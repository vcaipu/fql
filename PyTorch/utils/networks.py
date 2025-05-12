import math
from typing import Any, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform


class DefaultInit:
    """Default weight initialization class that's picklable."""
    def __init__(self, scale=1.0):
        self.scale = scale
        
    def __call__(self, tensor):
        return nn.init.orthogonal_(tensor, gain=self.scale)

def default_init(scale=1.0):
    """Default weight initialization."""
    return DefaultInit(scale)


class Identity(nn.Module):
    """Identity layer."""

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        input_dim: int = None,  # Required for PyTorch
        activations: Any = F.gelu,
        activate_final: bool = False,
        kernel_init: Any = default_init(),
        layer_norm: bool = False,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activations = activations
        self.activate_final = activate_final
        self.kernel_init = kernel_init
        self.layer_norm = layer_norm

        # If input_dim is not given, we'll set it during the forward pass
        self.input_dim = input_dim
        self._built = False
        self.layers = None
        self.layer_norms = None
        self.feature = None

    def build(self, input_dim):
        """Build the MLP layers."""
        self.input_dim = input_dim
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        prev_dim = input_dim
        for i, size in enumerate(self.hidden_dims):
            layer = nn.Linear(prev_dim, size)
            self.kernel_init(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
            if self.layer_norm and (i + 1 < len(self.hidden_dims) or self.activate_final):
                self.layer_norms.append(nn.LayerNorm(size))
            else:
                self.layer_norms.append(nn.Identity())
            prev_dim = size

        self._built = True

    def forward(self, x):
        device = x.device
        
        if not self._built:
            self.build(x.shape[-1])
            self.to(device)

        for i, (layer, layer_norm) in enumerate(zip(self.layers, self.layer_norms)):
            if layer.weight.device != device:
                layer.to(device)
            if hasattr(layer_norm, 'weight') and layer_norm.weight is not None and layer_norm.weight.device != device:
                layer_norm.to(device)
                
            x = layer(x)
            if i + 1 < len(self.layers) or self.activate_final:
                x = self.activations(x)
                x = layer_norm(x)
                if i == len(self.layers) - 2:
                    self.feature = x.detach()  # Store the feature for later use
        return x


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.log_value = nn.Parameter(torch.tensor(math.log(init_value)))

    def forward(self):
        return torch.exp(self.log_value)


class TransformedWithMode(TransformedDistribution):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.transforms[0](self.base_dist.mode)


class Actor(nn.Module):
    """Gaussian actor network.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        encoder: Optional encoder module to encode the inputs.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        action_dim: int,
        layer_norm: bool = False,
        log_std_min: Optional[float] = -5,
        log_std_max: Optional[float] = 2,
        tanh_squash: bool = False,
        state_dependent_std: bool = False,
        const_std: bool = True,
        final_fc_init_scale: float = 1e-2,
        encoder: nn.Module = None,
    ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.layer_norm = layer_norm
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.tanh_squash = tanh_squash
        self.state_dependent_std = state_dependent_std
        self.const_std = const_std
        self.final_fc_init_scale = final_fc_init_scale
        self.encoder = encoder

        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Linear(self.hidden_dims[-1], self.action_dim)
        nn.init.orthogonal_(self.mean_net.weight, gain=self.final_fc_init_scale)
        nn.init.zeros_(self.mean_net.bias)

        if self.state_dependent_std:
            self.log_std_net = nn.Linear(self.hidden_dims[-1], self.action_dim)
            nn.init.orthogonal_(self.log_std_net.weight, gain=self.final_fc_init_scale)
            nn.init.zeros_(self.log_std_net.bias)
        elif not self.const_std:
            self.log_stds = nn.Parameter(torch.zeros(self.action_dim))

    def forward(self, observations, temperature=1.0):
        """Return action distributions.

        Args:
            observations: Observations.
            temperature: Scaling factor for the standard deviation.
        """
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = torch.zeros_like(means)
            else:
                log_stds = self.log_stds.expand_as(means)

        log_stds = torch.clamp(log_stds, self.log_std_min, self.log_std_max)
        stds = torch.exp(log_stds) * temperature

        distribution = Normal(means, stds)
        if self.tanh_squash:
            transforms = [TanhTransform()]
            distribution = TransformedWithMode(distribution, transforms)

        return distribution


class EnsembleValue(nn.Module):
    """Ensemble of value networks."""

    def __init__(
        self,
        hidden_dims: Sequence[int],
        num_ensembles: int = 2,
        layer_norm: bool = True,
        encoder: nn.Module = None,
    ):
        super().__init__()
        self.ensemble = nn.ModuleList([
            Value(hidden_dims, layer_norm=layer_norm, encoder=encoder)
            for _ in range(num_ensembles)
        ])

    def forward(self, observations, actions=None):
        device = observations.device
        
        for net in self.ensemble:
            if next(net.parameters(), torch.empty(0)).device != device:
                net.to(device)
                
        return torch.stack([net(observations, actions) for net in self.ensemble])


class Value(nn.Module):
    """Value/critic network.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        layer_norm: bool = True,
        encoder: nn.Module = None,
    ):
        super().__init__()
        self.encoder = encoder
        # Initialize the MLP with its input dimension to ensure parameters are created
        self.value_net = MLP((*hidden_dims, 1), activate_final=False, layer_norm=layer_norm)
        # Create a dummy parameter to ensure parameters exist
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, observations, actions=None):
        """Return values or critic values.

        Args:
            observations: Observations.
            actions: Actions (optional).
        """
        device = observations.device
        
        if next(self.parameters(), torch.empty(0)).device != device:
            self.to(device)
            
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            if actions.device != device:
                actions = actions.to(device)
            inputs.append(actions)
        inputs = torch.cat(inputs, dim=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    def __init__(
        self,
        hidden_dims: Sequence[int],
        action_dim: int,
        layer_norm: bool = False,
        encoder: nn.Module = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_dim
        self.mlp = MLP((*hidden_dims, action_dim), activate_final=False, layer_norm=layer_norm)
        # Add a dummy parameter
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional)."""
        device = observations.device
        
        if next(self.parameters(), torch.empty(0)).device != device:
            self.to(device)
            
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            if actions.device != device:
                actions = actions.to(device)
            inputs = torch.cat([observations, actions], dim=-1)
        else:
            if actions.device != device:
                actions = actions.to(device)
            if times.device != device:
                times = times.to(device)
            inputs = torch.cat([observations, actions, times], dim=-1)

        v = self.mlp(inputs)
        return v
