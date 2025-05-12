import ogbench
import gymnasium


def get_base_env_name(env_name):
    """Get the base environment name from a singletask environment name."""
    if 'singletask' not in env_name:
        return env_name
    
    # The format should be like 'antsoccer-arena-navigate-singletask-v0'
    # or 'antsoccer-arena-navigate-singletask-task1-v0'
    parts = env_name.split('-')
    
    # Find the position of 'singletask'
    try:
        singletask_idx = parts.index('singletask')
    except ValueError:
        for i, part in enumerate(parts):
            if 'singletask' in part:
                singletask_idx = i
                break
        else:
            return env_name
    
    # The base name is everything before 'singletask'
    base_parts = parts[:singletask_idx]
    
    # If there are version parts after 'singletask', add them back
    for part in parts[singletask_idx+1:]:
        if part.startswith('v') and part[1:].isdigit():
            base_parts.append(part)
            break
        elif part.startswith('task') and 'v' in part:
            # Handle 'task1-v0' format
            task_version = part.split('-')
            if len(task_version) > 1:
                version = task_version[1]
                if version.startswith('v') and version[1:].isdigit():
                    base_parts.append(version)
            break
    
    # Join back into a base environment name
    return '-'.join(base_parts)


def make_env_and_datasets(env_name, env_only=False):
    """Make OGBench environment and datasets with render_mode=None."""
    base_env_name = get_base_env_name(env_name)
    
    # Try creating the environment with render_mode=None
    try:
        env = gymnasium.make(base_env_name, render_mode=None)
    except (TypeError, ValueError) as e:
        print(f"Warning: Couldn't create environment with render_mode=None: {e}")
        try:
            env = gymnasium.make(base_env_name)
            # Try to disable rendering if possible
            if hasattr(env.unwrapped, 'render_mode'):
                env.unwrapped.render_mode = None
        except Exception as e2:
            print(f"Error creating environment: {e2}")
            # As a last resort, try the original env_name
            env = gymnasium.make(env_name)
    
    # Wrap the environment for singletask if needed
    if 'singletask' in env_name:
        try:
            env = ogbench.wrap_to_singletask(env, env_name)
        except Exception:
            env = ogbench.wrap_to_singletask(env)
    
    if env_only:
        return env
    
    try:
        datasets = ogbench.get_datasets(env_name)
        train_dataset, val_dataset = datasets
    except Exception as e:
        print(f"Error getting datasets: {e}")
        try:
            datasets = ogbench.get_datasets(f"{base_env_name}-singletask")
            train_dataset, val_dataset = datasets
        except Exception:
            raise ValueError(f"Failed to get datasets for {env_name}")
    
    return env, train_dataset, val_dataset
