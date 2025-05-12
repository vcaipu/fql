module purge
module load cudatoolkit/12.8 
source fqlorig/bin/activate
export MUJOCO_GL=osmesa
python3 main.py --env_name=antsoccer-arena-navigate-singletask-v0 --agent.discount=0.995 --agent.alpha=10