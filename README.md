# Game Agent Mario Evaluation

Standalone evaluation code for visual-language game agents on:

- Super Mario Bros with color observations
- Super Mario Land with black-white observations

The main entry point is `game_agent_evaluation_fromuser.py`. It loads a game environment config, an interaction model config, runs the agent for a fixed number of steps, and writes per-step interaction records to a user-specified `--log_dir`.

## Code Structure

```text
.
├── README.md
├── __init__.py
├── game_agent_evaluation_fromuser.py
├── utils.py
├── configs/
│   ├── super_mario_bros/
│   └── super_mario_land/
├── game_envs/
│   ├── super_mario_bros/
│   └── super_mario_land/
├── model_configs/
└── slurm/
    └── example_scripts/
```

## Main Files

- `game_agent_evaluation_fromuser.py`: evaluation runner. It creates the environment, sends observations to the model, parses button actions, steps the emulator, and saves interaction traces.
- `utils.py`: helper functions for button parsing, class lookup, and image resizing.
- `configs/`: game-level configuration files for the supported evaluation settings.
- `game_envs/`: local Super Mario environment implementations used by the evaluator.
- `model_configs/`: interaction model config templates.
- `slurm/example_scripts/`: template Slurm launch scripts for batch evaluation.

## Game Configs

Super Mario Bros configs live in:

```text
configs/super_mario_bros/
```

Super Mario Land configs live in:

```text
configs/super_mario_land/
```

Pass one config to the evaluator with `--game_config_path`.

## Model Configs

The release includes template model configs:

```text
model_configs/interaction_model_config_qwen3-8b-base.yaml
model_configs/interaction_model_config_qwen3-8b-rl-base.yaml
model_configs/interaction_model_config_qwen3-8b-rl-sft.yaml
```

Before running an evaluation, fill in at least:

```yaml
model_name: <model name or checkpoint path>
base_url: http://localhost:<port>/v1/
```

The evaluator expects an OpenAI-compatible chat/completions endpoint. The Slurm examples start a local vLLM server and then point the model config at that server.

## Slurm Examples

Template launch scripts are in:

```text
slurm/example_scripts/
```

Available templates:

```text
slurm/example_scripts/smb.sh
slurm/example_scripts/sml.sh
```

Edit the placeholders in these scripts for your cluster and model setup: `EVAL_DIR`, `MODEL_NAME`, `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, output log paths, `interaction_model`, `MAX_PARALLEL_JOBS`.

## Outputs

Each run writes interaction JSON files and any configured visual artifacts under `--log_dir`. A typical log directory is nested by game, world, level, start state, model name, and run id so multiple evaluations can be compared after completion.
