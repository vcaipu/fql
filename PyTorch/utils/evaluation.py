from collections import defaultdict
import traceback

import numpy as np
import torch
from tqdm import trange


def supply_rng(f, rng=None):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        if rng is None:
            seed = torch.randint(0, 2**32, (1,)).item()
            return f(*args, seed=seed, **kwargs)
        else:
            return f(*args, seed=rng, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    agent,
    env,
    config=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
):
    """Evaluate the agent in the environment.

    Args:
        agent: Agent.
        env: Environment.
        config: Configuration dictionary.
        num_eval_episodes: Number of episodes to evaluate the agent.
        num_video_episodes: Number of episodes to render. These episodes are not included in the statistics.
        video_frame_skip: Number of frames to skip between renders.
        eval_temperature: Action sampling temperature.

    Returns:
        A tuple containing the statistics, trajectories, and rendered videos.
    """
    try:
        device = next(agent.network.model_def.parameters()).device
        
        actor_fn = supply_rng(agent.sample_actions, rng=torch.randint(0, 2**32, (1,)).item())
        trajs = []
        stats = defaultdict(list)

        dummy_stats = {'episode.return': 0.0, 'episode.length': 0}

        disable_rendering = True
        try:
            test_frame = env.render()
            if test_frame is not None:
                disable_rendering = False
        except:
            print("Rendering disabled due to initialization error")
        
        renders = []
        
        for i in trange(num_eval_episodes + num_video_episodes):
            traj = defaultdict(list)
            should_render = i >= num_eval_episodes and not disable_rendering and num_video_episodes > 0

            try:
                observation, info = env.reset()
            except Exception as e:
                print(f"Environment reset failed: {e}")
                return dummy_stats, [], []
            
            done = False
            step = 0
            render = []
            
            try:
                while not done:
                    if isinstance(observation, np.ndarray):
                        observation_tensor = torch.from_numpy(observation).float().to(device)
                    else:
                        observation_tensor = observation.to(device)
                    
                    action = actor_fn(observations=observation_tensor, temperature=eval_temperature)
                    
                    if isinstance(action, torch.Tensor):
                        action = action.detach().cpu().numpy()
                    action = np.clip(action, -1, 1)

                    try:
                        next_observation, reward, terminated, truncated, info = env.step(action)
                    except Exception as e:
                        print(f"Environment step failed: {e}")
                        break
                        
                    done = terminated or truncated
                    step += 1

                    if should_render and (step % video_frame_skip == 0 or done):
                        try:
                            frame = env.render()
                            if frame is not None:
                                render.append(frame.copy())
                        except Exception as e:
                            print(f"Rendering failed: {e}")
                            disable_rendering = True
                            should_render = False
                            render = []

                    transition = dict(
                        observation=observation,
                        next_observation=next_observation,
                        action=action,
                        reward=reward,
                        done=done,
                        info=info,
                    )
                    add_to(traj, transition)
                    observation = next_observation
                
                if i < num_eval_episodes:
                    add_to(stats, flatten(info))
                    trajs.append(traj)
                elif len(render) > 0:
                    renders.append(np.array(render))
            
            except Exception as e:
                print(f"Error during evaluation: {e}")
                print(traceback.format_exc())
                continue

        # Calculate statistics
        if stats:
            stats_dict = {k: np.mean(v) for k, v in stats.items()}
        else:
            stats_dict = dummy_stats
            
        return stats_dict, trajs, renders
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        print(traceback.format_exc())
        return {'episode.return': 0.0, 'episode.length': 0}, [], []
