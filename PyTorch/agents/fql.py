import copy
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam

from utils.networks import ActorVectorField, EnsembleValue
from utils.encoders import encoder_modules
from utils.torch_utils import ModuleDict, TrainState


class FQLAgent:
    """Flow Q-learning (FQL) agent."""

    def __init__(
        self,
        network: TrainState,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ):
        self.network = network
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def critic_loss(self, batch, grad_params=None):
        """Compute the FQL critic loss."""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.no_grad():
            next_action_noise = torch.randn(
                (batch['next_observations'].shape[0], self.config['action_dim']),
                device=self.device
            )
            next_actions = self.sample_actions(batch['next_observations'], seed=next_action_noise)
            next_actions = torch.clamp(next_actions, -1, 1)

            next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)
            if self.config['q_agg'] == 'min':
                next_q = next_qs.min(dim=0)[0]
            else:
                next_q = next_qs.mean(dim=0)

            target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q

        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = F.mse_loss(q, target_q.expand_as(q))

        info = {
            'critic_loss': critic_loss.item(),
            'q_mean': q.mean().item(),
            'q_max': q.max().item(),
            'q_min': q.min().item(),
        }

        return critic_loss, info

    def actor_loss(self, batch, grad_params=None):
        """Compute the FQL actor loss."""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_size, action_dim = batch['actions'].shape

        # BC flow loss.
        x_0 = torch.randn((batch_size, action_dim), device=self.device)
        x_1 = batch['actions']
        t = torch.rand((batch_size, 1), device=self.device)
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_bc_flow')(batch['observations'], x_t, t, params=grad_params)
        bc_flow_loss = torch.mean((pred - vel) ** 2)

        # Distillation loss.
        noises = torch.randn((batch_size, action_dim), device=self.device)
        with torch.no_grad():
            target_flow_actions = self.compute_flow_actions(batch['observations'], noises=noises)
        actor_actions = self.network.select('actor_onestep_flow')(batch['observations'], noises, params=grad_params)
        distill_loss = torch.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        actor_actions = torch.clamp(actor_actions, -1, 1)
        qs = self.network.select('critic')(batch['observations'], actions=actor_actions)
        q = torch.mean(qs, dim=0)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = 1 / torch.abs(q).mean().detach()
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha'] * distill_loss + q_loss

        # Additional metrics for logging.
        with torch.no_grad():
            actions = self.sample_actions(batch['observations'])
            mse = torch.mean((actions - batch['actions']) ** 2)

        info = {
            'actor_loss': actor_loss.item(),
            'bc_flow_loss': bc_flow_loss.item(),
            'distill_loss': distill_loss.item(),
            'q_loss': q_loss.item(),
            'q': q.mean().item(),
            'mse': mse.item(),
        }

        return actor_loss, info

    def total_loss(self, batch, grad_params=None):
        """Compute the total loss."""
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        info = {}

        critic_loss, critic_info = self.critic_loss(batch, grad_params)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self):
        """Update the target network."""
        for target_param, param in zip(
            self.network.model_def.modules_dict['target_critic'].parameters(),
            self.network.model_def.modules_dict['critic'].parameters()
        ):
            target_param.data.copy_(
                param.data * self.config['tau'] + target_param.data * (1 - self.config['tau'])
            )

    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        def loss_fn():
            return self.total_loss(batch)

        new_network, info = self.network.apply_loss_fn(loss_fn)
        self.target_update()

        return self, info

    def sample_actions(self, observations, temperature=1.0, seed=None):
        """Sample actions from the one-step policy."""
        if isinstance(observations, torch.Tensor):
            observations = observations.to(self.device)
        else:
            observations = torch.tensor(observations, dtype=torch.float32, device=self.device)
            
        # Generate noise
        batch_shape = observations.shape[:-len(self.config['ob_dims'])] if hasattr(self.config, 'ob_dims') else observations.shape[:-1]
        if seed is None:
            noises = torch.randn((*batch_shape, self.config['action_dim']), device=self.device)
        else:
            if isinstance(seed, torch.Tensor):
                noises = seed
                if noises.device != self.device:
                    noises = noises.to(self.device)
            else:
                # Use the seed to set the random state
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
                noises = torch.randn((*batch_shape, self.config['action_dim']), 
                                     device=self.device, 
                                     generator=generator)
            
        # Get actions from one-step policy
        with torch.no_grad():
            actions = self.network.select('actor_onestep_flow')(observations, noises)
            actions = torch.clamp(actions, -1, 1)
            
        return actions

    def compute_flow_actions(self, observations, noises):
        """Compute actions from the BC flow model using the Euler method."""
        if isinstance(observations, torch.Tensor) and observations.device != self.device:
            observations = observations.to(self.device)
        if isinstance(noises, torch.Tensor) and noises.device != self.device:
            noises = noises.to(self.device)
                
        # Apply encoder if needed
        if 'encoder' in self.config and self.config['encoder'] is not None:
            with torch.no_grad():
                if hasattr(self.network.model_def.modules_dict, 'actor_bc_flow_encoder'):
                    encoded_obs = self.network.select('actor_bc_flow_encoder')(observations)
                else:
                    # Encode observations within the flow network
                    encoded_obs = observations
        else:
            encoded_obs = observations
            
        # Euler method
        with torch.no_grad():
            actions = noises
            for i in range(self.config['flow_steps']):
                t = torch.full((*observations.shape[:-1], 1), i / self.config['flow_steps'], device=self.device)
                vels = self.network.select('actor_bc_flow')(encoded_obs, actions, t, is_encoded=True)
                actions = actions + vels / self.config['flow_steps']
                
            actions = torch.clamp(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: Any,
        ex_actions: Any,
        config: Dict[str, Any],
    ):
        """Create a new agent."""
        torch.manual_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not isinstance(ex_observations, torch.Tensor):
            ex_observations = torch.tensor(ex_observations, dtype=torch.float32)
        if not isinstance(ex_actions, torch.Tensor):
            ex_actions = torch.tensor(ex_actions, dtype=torch.float32)

        ex_observations = ex_observations.to(device)
        ex_actions = ex_actions.to(device)
        
        # Create time values (0 to 1)
        batch_size = ex_observations.shape[0] if len(ex_observations.shape) > 1 else 1
        ex_times = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        
        ob_dims = tuple(ex_observations.shape[1:])
        action_dim = ex_actions.shape[-1]

        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module().to(device)
            encoders['actor_bc_flow'] = encoder_module().to(device)
            encoders['actor_onestep_flow'] = encoder_module().to(device)

        critic_def = EnsembleValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        ).to(device)
        
        actor_bc_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_flow'),
        ).to(device)
        
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_onestep_flow'),
        ).to(device)

        # Create ModuleDict with all networks
        networks = {
            'critic': critic_def,
            'target_critic': copy.deepcopy(critic_def).to(device),
            'actor_bc_flow': actor_bc_flow_def,
            'actor_onestep_flow': actor_onestep_flow_def,
        }
        
        if encoders.get('actor_bc_flow') is not None:
            # Add actor_bc_flow_encoder to ModuleDict to make it separately callable.
            networks['actor_bc_flow_encoder'] = encoders.get('actor_bc_flow')
                
        network_def = ModuleDict(networks)
        network_def.to(device)
        
        # Initialize optimizer
        network_tx = lambda params: Adam(params, lr=config['lr'])
        
        # Create train state
        network = TrainState.create(network_def, tx=network_tx)

        # Add additional config parameters
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        
        return cls(network=network, config=config, device=device)


def get_config():
    """Get the default configuration for FQL."""
    config = {
        'agent_name': 'fql',  # Agent name.
        'ob_dims': None,  # Observation dimensions (will be set automatically).
        'action_dim': None,  # Action dimension (will be set automatically).
        'lr': 3e-4,  # Learning rate.
        'batch_size': 256,  # Batch size.
        'actor_hidden_dims': (512, 512, 512, 512),  # Actor network hidden dimensions.
        'value_hidden_dims': (512, 512, 512, 512),  # Value network hidden dimensions.
        'layer_norm': True,  # Whether to use layer normalization.
        'actor_layer_norm': False,  # Whether to use layer normalization for the actor.
        'discount': 0.99,  # Discount factor.
        'tau': 0.005,  # Target network update rate.
        'q_agg': 'mean',  # Aggregation method for target Q values.
        'alpha': 10.0,  # BC coefficient (need to be tuned for each environment).
        'flow_steps': 10,  # Number of flow steps.
        'normalize_q_loss': False,  # Whether to normalize the Q loss.
        'encoder': None,  # Visual encoder name (None, 'impala_small', etc.).
    }
    
    return config
