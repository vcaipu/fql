import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence, Union, Callable

import torch
import torch.nn as nn
import torch.optim as optim


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    def __init__(self, modules: Dict[str, nn.Module]):
        super().__init__()
        self.modules_dict = nn.ModuleDict({k: v for k, v in modules.items()})

    def forward(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if set(kwargs.keys()) != set(self.modules_dict.keys()):
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules_dict.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                module = self.modules_dict[key]
                if isinstance(value, Mapping):
                    out[key] = module(**value)
                elif isinstance(value, Sequence) and not isinstance(value, (str, torch.Tensor)):
                    out[key] = module(*value)
                else:
                    out[key] = module(value)
            return out

        return self.modules_dict[name](*args, **kwargs)


class TrainState:
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optimizer.
        opt_state: Optimizer state.
    """

    def __init__(
        self,
        step: int,
        model_def: nn.Module,
        optimizer: optim.Optimizer,
        **kwargs
    ):
        self.step = step
        self.model_def = model_def
        self.optimizer = optimizer
        self.__dict__.update(kwargs)

    @classmethod
    def create(cls, model_def, params=None, tx=None, **kwargs):
        """Create a new train state."""
        # Initialize the model with dummy inputs if needed
        if hasattr(kwargs, 'network_args'):
            # Forward pass to initialize parameters
            model_def(**kwargs['network_args'])
        
        # Ensure the model is properly initialized
        if next(model_def.parameters(), None) is None:
            raise ValueError("Model has no parameters. Make sure it's properly initialized.")
            
        if tx is not None:
            optimizer = tx(model_def.parameters())
        else:
            optimizer = None

        return cls(
            step=1,
            model_def=model_def,
            optimizer=optimizer,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters.
            method: Method to call in the model. If `None`, it uses the default `forward` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if method is not None:
            method_name = getattr(self.model_def, method)
            return method_name(*args, **kwargs)
        else:
            return self.model_def(*args, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, **kwargs):
        """Apply the gradients and return the updated state."""
        self.optimizer.step()
        self.step += 1
        
        # Update other attributes if provided
        self.__dict__.update(kwargs)
        
        return self

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info."""
        self.optimizer.zero_grad()
        loss, info = loss_fn()
        loss.backward()
        
        # Compute gradient statistics
        grad_info = {}
        grad_max_list = []
        grad_min_list = []
        grad_norm_list = []
        
        for name, param in self.model_def.named_parameters():
            if param.grad is not None:
                grad_max = param.grad.max().item()
                grad_min = param.grad.min().item()
                grad_norm = param.grad.norm(p=2).item()
                
                grad_max_list.append(grad_max)
                grad_min_list.append(grad_min)
                grad_norm_list.append(grad_norm)
        
        if grad_max_list:
            final_grad_max = max(grad_max_list)
            final_grad_min = min(grad_min_list)
            final_grad_norm = sum(grad_norm_list)
            
            grad_info = {
                'grad/max': final_grad_max,
                'grad/min': final_grad_min,
                'grad/norm': final_grad_norm,
            }
            
        info.update(grad_info)
        
        # Apply gradients
        self.optimizer.step()
        self.step += 1
        
        return self, info


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'params_{epoch}.pt')
    
    # Create a state dictionary with all the necessary components
    state_dict = {
        'network_params': agent.network.model_def.state_dict(),
        'optimizer_state': agent.network.optimizer.state_dict(),
        'config': agent.config,
        'step': agent.network.step
    }
    
    torch.save(state_dict, save_path)
    
    print(f'Saved to {save_path}')


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)
    
    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'
    
    restore_path = os.path.join(candidates[0], f'params_{restore_epoch}.pt')
    
    checkpoint = torch.load(restore_path)
    
    # Load the state dictionary components
    agent.network.model_def.load_state_dict(checkpoint['network_params'])
    agent.network.optimizer.load_state_dict(checkpoint['optimizer_state'])
    agent.config.update(checkpoint['config'])
    agent.network.step = checkpoint['step']
    
    print(f'Restored from {restore_path}')
    
    return agent
