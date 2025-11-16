# Repository Guidelines

## Project Structure & Module Organization
Core models sit inside `models/` (shared blocks in `models/common/`), while data interfaces live in `dataloaders/`. Training pipelines are in `training/` plus the entry scripts `train.py`, `trainer.py`, and `self_feed.py`. Config defaults stay in `config.yaml` and helper variants under `utils/config_models.py`. Store datasets or synthetic seeds in `datasets/`, and treat `runs/`, `wandb/`, and `saved_simulations/` as disposable artifact folders when iterating.

## Runtime
Use container as a runtime:
docker run --user $(id -u):$(id -g) --env-file .env --gpus all -it -v $(pwd):/n_body_approx nbody-cuda