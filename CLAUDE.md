# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow Rules

1. First think through the problem, read the codebase for relevant files, and write a plan to todo.md.
2. The plan should have a list of todo items that you can check off as you complete them
3. Before you begin working, check in with me and I will verify the plan.
4. Then, begin working on the todo items, marking them as complete as you go.
5. Please every step of the way just give me a high level explanation of what changes you made
6. Make every task and code change you do as simple as possible. We want to avoid making any massive or complex changes. Every change should impact as little code as possible. Everything is about simplicity.
7. Finally, add a review section to the todo.md file with a summary of the changes you made and any other relevant information.

## Project Overview

This is a specialized Isaac Lab project focused on reinforcement learning with mimic functionality. It's built on top of NVIDIA Isaac Lab, a GPU-accelerated robotics framework for RL, imitation learning, and motion planning.

## Common Development Commands

### Training
```bash
# Basic training
python train.py --task franka_lift --num_envs 256 --headless

# Training with specific seeds and wandb logging
python train.py --task franka_lift --num_envs 256 --seed 42 --headless --project wandb_project_name

# Resume training from checkpoint
python train.py --task franka_lift --num_envs 256 --headless --resume --checkpoint path/to/checkpoint.pt
```

### Evaluation
```bash
# Basic evaluation
python play.py --task franka_lift --checkpoint path/to/checkpoint.pt

# Evaluation with specific number of episodes
python play.py --task franka_lift --checkpoint path/to/checkpoint.pt --num_episodes 100
```

### Code Quality
```bash
# Format code (Black, isort, autoflake)
make format

# Run linting (mypy, flake8)
make lint

# Run specific formatters
python -m black .
python -m isort .
python -m autoflake --in-place --remove-all-unused-imports --recursive .
```

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_example.py

# Run with coverage
python -m pytest --cov=isaaclab_rl_project_mimic
```

## High-Level Architecture

### Project Structure
```
isaaclab_rl_project_mimic/
├── tasks/               # Task definitions
│   ├── franka/         # Franka robot tasks (e.g., lifting)
│   ├── mimic/          # Mimic-based tasks
│   └── template/       # Template for new tasks
├── configs/            # Hydra configuration files
├── checkpoints/        # Saved model checkpoints
├── agent_cfgs/         # RL agent configurations
├── train.py           # Main training script
├── play.py           # Evaluation script
└── sweep.py          # Hyperparameter sweeping
```

### Key Components

1. **Task System**: Tasks are defined in `tasks/` directory. Each task includes:
   - Environment configuration (`env_cfg.py`)
   - Task logic implementation
   - Registration with the task registry

2. **Configuration**: Uses Hydra for configuration management
   - Base configs in `configs/`
   - Override with command-line arguments
   - Supports configuration composition

3. **RL Framework Integration**: 
   - Primary framework: skrl (default)
   - Supports multiple backends via `--library` flag
   - Agent configs in `agent_cfgs/`

4. **Observation/Action Spaces**:
   - Defined per task in environment configs
   - Supports multi-modal observations
   - Configurable action spaces

## Available Tasks

Current tasks in this project:
- `franka_lift`: Franka robot lifting task
- `franka_mimic`: Mimic-based Franka task
- `template_task`: Template for creating new tasks

## Development Workflow

1. **Creating New Tasks**:
   - Copy template from `tasks/template/`
   - Define environment configuration
   - Implement task logic
   - Register in `__init__.py`

2. **Modifying Existing Tasks**:
   - Environment parameters in `env_cfg.py`
   - Task logic in main task file
   - Observation/action spaces in configs

3. **Training Experiments**:
   - Use wandb for tracking: `--project project_name`
   - Sweep hyperparameters with `sweep.py`
   - Save checkpoints automatically

4. **Debugging**:
   - Remove `--headless` to see visualization
   - Use `--debug` flag for additional logging
   - Check `logs/` directory for detailed outputs

## Important Notes

- Always run from the project root directory
- Use `--headless` for faster training without GUI
- Checkpoint paths are relative to project root
- Default device is CUDA if available
- Environment configs use SI units

## Integration with Main Isaac Lab

This project extends the main Isaac Lab framework located at `/home/simon/IsaacLab/`. For core framework changes or to access additional environments, refer to the main Isaac Lab directory.

Main Isaac Lab commands:
```bash
# Install/update Isaac Lab
./isaaclab.sh --install

# Run with Isaac Sim
./isaaclab.sh -p train.py --task franka_lift

# Format entire Isaac Lab codebase
./isaaclab.sh -p source/standalone/tools/run_formatter.py --python
```