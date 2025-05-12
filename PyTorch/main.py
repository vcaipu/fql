import json
import random
import time
import argparse

import numpy as np
import torch
import tqdm
import wandb
from ml_collections import ConfigDict

from agents import agents
from envs.env_utils import make_env_and_datasets
from utils.datasets import Dataset, ReplayBuffer
from utils.evaluation import evaluate, flatten
from utils.torch_utils import save_agent, restore_agent
from utils.log_utils import CsvLogger, get_exp_name, setup_wandb, get_wandb_video

# Create argument parser
parser = argparse.ArgumentParser(description='FQL: Flow Q-Learning for Continuous Control')
parser.add_argument('--run_group', type=str, default='Debug', help='Run group name')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--env_name', type=str, default='cube-double-play-singletask-v0', help='Environment name')
parser.add_argument('--save_dir', type=str, default='exp/', help='Save directory')
parser.add_argument('--restore_path', type=str, default=None, help='Restore path')
parser.add_argument('--restore_epoch', type=int, default=None, help='Restore epoch')
parser.add_argument('--offline_steps', type=int, default=1000000, help='Number of offline training steps')
parser.add_argument('--online_steps', type=int, default=0, help='Number of online training steps')
parser.add_argument('--buffer_size', type=int, default=2000000, help='Replay buffer size')
parser.add_argument('--log_interval', type=int, default=5000, help='Logging interval')
parser.add_argument('--eval_interval', type=int, default=10000, help='Evaluation interval')
parser.add_argument('--save_interval', type=int, default=100000, help='Saving interval')
parser.add_argument('--eval_episodes', type=int, default=50, help='Number of evaluation episodes')
parser.add_argument('--video_episodes', type=int, default=0, help='Number of video episodes for each task')
parser.add_argument('--video_frame_skip', type=int, default=3, help='Frame skip for videos')
parser.add_argument('--p_aug', type=float, default=None, help='Probability of applying image augmentation')
parser.add_argument('--frame_stack', type=int, default=None, help='Number of frames to stack')
parser.add_argument('--balanced_sampling', type=int, default=0, help='Whether to use balanced sampling for online fine-tuning')

# Agent configuration
parser.add_argument('--agent_name', type=str, default='fql', help='Agent name')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
parser.add_argument('--tau', type=float, default=0.005, help='Target network update rate')
parser.add_argument('--q_agg', type=str, default='mean', help='Aggregation method for target Q values')
parser.add_argument('--alpha', type=float, default=10.0, help='BC coefficient')
parser.add_argument('--flow_steps', type=int, default=10, help='Number of flow steps')
parser.add_argument('--normalize_q_loss', type=int, default=0, help='Whether to normalize the Q loss')
parser.add_argument('--encoder', type=str, default=None, help='Visual encoder name')


def main():
    # Configure environment variables for headless rendering
    import os
    os.environ['MUJOCO_GL'] = 'egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logger
    exp_name = get_exp_name(args.seed)
    run = setup_wandb(project='fql_pytorch', group=args.run_group, name=exp_name, config=vars(args))

    # Create save directory
    args.save_dir = os.path.join(args.save_dir, 'fql_pytorch', args.run_group, exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Create configuration dictionary
    config = ConfigDict({
        'agent_name': args.agent_name,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'actor_hidden_dims': (512, 512, 512, 512),
        'value_hidden_dims': (512, 512, 512, 512),
        'layer_norm': True,
        'actor_layer_norm': False,
        'discount': args.discount,
        'tau': args.tau,
        'q_agg': args.q_agg,
        'alpha': args.alpha,
        'flow_steps': args.flow_steps,
        'normalize_q_loss': bool(args.normalize_q_loss),
        'encoder': args.encoder,
    })

    # Make environment and datasets
    env, eval_env, train_dataset, val_dataset = make_env_and_datasets(
        args.env_name, frame_stack=args.frame_stack
    )
    if args.video_episodes > 0:
        assert 'singletask' in args.env_name, 'Rendering is currently only supported for OGBench environments.'
    if args.online_steps > 0:
        assert 'visual' not in args.env_name, 'Online fine-tuning is currently not supported for visual environments.'

    # Check that we have valid datasets
    if train_dataset is None or train_dataset.size == 0:
        raise ValueError(f"Failed to load dataset for {args.env_name}")
    
    print(f"Dataset size: {train_dataset.size}")
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Set up datasets
    if args.balanced_sampling:
        # Create a separate replay buffer so that we can sample from both the training dataset and the replay buffer
        example_transition = {k: v[0:1] for k, v in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(example_transition, size=args.buffer_size)
    else:
        # Use the training dataset as the replay buffer
        replay_buffer = ReplayBuffer.create_from_initial_dataset(
            {k: v for k, v in train_dataset.items()}, size=max(args.buffer_size, train_dataset.size + 1)
        )
        train_dataset = replay_buffer

    # Set p_aug and frame_stack
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = args.p_aug
            dataset.frame_stack = args.frame_stack

    # Create agent
    example_batch = train_dataset.sample(1)
    agent_class = agents[config.agent_name]
    agent = agent_class.create(
        args.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )

    # Restore agent if needed
    if args.restore_path is not None:
        agent = restore_agent(agent, args.restore_path, args.restore_epoch)

    # Train agent
    train_logger = CsvLogger(os.path.join(args.save_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(args.save_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()

    step = 0
    done = True
    expl_metrics = dict()
    
    for i in tqdm.tqdm(range(1, args.offline_steps + args.online_steps + 1), smoothing=0.1, dynamic_ncols=True):
        if i <= args.offline_steps:
            # Offline RL
            batch = train_dataset.sample(config.batch_size)
            agent, update_info = agent.update(batch)
        else:
            # Online fine-tuning
            if done:
                step = 0
                ob, _ = env.reset()

            # Sample action from agent
            if isinstance(ob, np.ndarray):
                ob_tensor = torch.from_numpy(ob).float().to(agent.device)
            else:
                ob_tensor = ob.to(agent.device)
                
            action = agent.sample_actions(observations=ob_tensor)
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            
            # Step environment
            next_ob, reward, terminated, truncated, info = env.step(action.copy())
            done = terminated or truncated

            # Adjust reward for D4RL antmaze
            if 'antmaze' in args.env_name and (
                'diverse' in args.env_name or 'play' in args.env_name or 'umaze' in args.env_name
            ):
                reward = reward - 1.0

            # Add transition to replay buffer
            replay_buffer.add_transition(
                dict(
                    observations=ob,
                    actions=action,
                    rewards=reward,
                    terminals=float(done),
                    masks=1.0 - terminated,
                    next_observations=next_ob,
                )
            )
            ob = next_ob

            if done:
                expl_metrics = {f'exploration/{k}': np.mean(v) for k, v in flatten(info).items()}

            step += 1

            # Update agent
            if args.balanced_sampling:
                # Half-and-half sampling from the training dataset and the replay buffer
                dataset_batch = train_dataset.sample(config.batch_size // 2)
                replay_batch = replay_buffer.sample(config.batch_size // 2)
                batch = {k: torch.cat([dataset_batch[k], replay_batch[k]], dim=0) for k in dataset_batch}
            else:
                batch = train_dataset.sample(config.batch_size)

            agent, update_info = agent.update(batch)

        # Log metrics
        if i % args.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            
            if val_dataset is not None:
                val_batch = val_dataset.sample(config.batch_size)
                _, val_info = agent.total_loss(val_batch)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
                
            train_metrics['time/epoch_time'] = (time.time() - last_time) / args.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            train_metrics.update(expl_metrics)
            last_time = time.time()
            
            wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        # Evaluate agent
        if args.eval_interval != 0 and (i == 1 or i % args.eval_interval == 0):
            renders = []
            eval_metrics = {}
            
            eval_info, trajs, cur_renders = evaluate(
                agent=agent,
                env=eval_env,
                config=config,
                num_eval_episodes=args.eval_episodes,
                num_video_episodes=args.video_episodes,
                video_frame_skip=args.video_frame_skip,
            )
            
            if cur_renders:
                renders.extend(cur_renders)
            for k, v in eval_info.items():
                eval_metrics[f'evaluation/{k}'] = v

            if args.video_episodes > 0 and renders:
                try:
                    video = get_wandb_video(renders=renders)
                    eval_metrics['video'] = video
                except Exception as e:
                    print(f"Failed to create video: {e}")

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

        # Save agent
        if i % args.save_interval == 0:
            save_agent(agent, args.save_dir, i)

    train_logger.close()
    eval_logger.close()


if __name__ == '__main__':
    main()
