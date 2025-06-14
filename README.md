# isaaclab_rl_project

![isaaclab_rl](https://github.com/user-attachments/assets/72036a2f-41ab-4317-ad30-8a165afa83a5)

## Installation 

### 1. Follow install instructions [here](https://github.com/elle-miller/isaaclab_rl)

### 2. Create your own project repo

Two options:

a) Make a fork (this will be public) to track upstream changes

b) Clone a private copy by creating a [new repository](https://github.com/new) (choose new name, do not initialise with readme etc)

```
git clone git@github.com:elle-miller/isaaclab_rl_project.git
mv isaaclab_rl_project my_cool_project_name
cd my_cool_project_name
git remote remove origin
git remote add origin git@github.com:yourusername/my_cool_project_name.git
git push -u origin main
```

### 3. Test everything is working OK
```
python train.py --task Franka_Lift --num_envs 8192 --headless

# play checkpoint with viewer
python play.py --task Franka_Lift --num_envs 256 --checkpoint logs/franka/lift/.../checkpoints/best_agent.pt
```
You should hit a return of ~8000 by 40 million timesteps (check "Eval episode returns / returns" on wandb)

### 4. Make your own environment

TODO
