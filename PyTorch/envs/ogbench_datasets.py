import ogbench
import numpy as np
import gymnasium
from ogbench.utils import make_env_and_datasets

def get_datasets(env_name):
    """Get OGBench datasets without creating environments."""
    # Call the full function but extract only the datasets
    try:
        _, train_dataset, val_dataset = make_env_and_datasets(env_name)
        return train_dataset, val_dataset
    except Exception as e:
        print(f"Error getting datasets: {e}")
        env = make_env_and_datasets(env_name, env_only=True)
        if hasattr(env, 'get_dataset'):
            dataset = env.get_dataset()
            n = len(dataset['observations'])
            split = int(0.9 * n)
            train_dataset = {k: v[:split] for k, v in dataset.items()}
            val_dataset = {k: v[split:] for k, v in dataset.items()}
            return train_dataset, val_dataset
        else:
            raise ValueError(f"Could not get dataset for {env_name}")

def get_wrapped_env(env_name, render_mode=None):
    """Get wrapped environment without rendering initialization."""
    try:
        env = make_env_and_datasets(env_name, env_only=True)
        return env
    except Exception as e:
        print(f"Error creating environment: {e}")
        raise
