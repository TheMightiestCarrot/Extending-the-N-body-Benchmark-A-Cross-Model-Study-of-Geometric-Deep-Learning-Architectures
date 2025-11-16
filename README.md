# n_body_approx

This document provides instructions on how to set up, train, and benchmark models on nbody simulations in self-feed mode.

## Installation

The easiest and most convenient way to install the project is to use Docker.

### Building the Docker image

```bash
docker build \
  --build-arg UID=$(id -u) \
  --build-arg GID=$(id -g) \
  -t nbody-cuda .
```

### Running the Docker container

```bash
# Interactive shell (no need for --user)
docker run --env-file .env --gpus all -it -v $(pwd):/n_body_approx nbody-cuda

# Or run a command directly, e.g. training
docker run --rm --env-file .env --gpus all -v $(pwd):/n_body_approx \
  nbody-cuda \
  python -m train --config config.yaml --trainer_type trainer_nbody \
  --model_type segnn --dataloader_type segnn_nbody \
  --trainer.learning_rate 1.0858181069399002 \
  --model.num_layers 6 --model.hidden_features 192 --model.lmax_h 1 \
  --trainer.steps_per_epoch 1000 --trainer.test_macros_every 10
```

## Training the Models

To train, run the training script with the desired parameters. Here is an example command:

```bash
python -m train --config config.yaml --trainer_type trainer_nbody --model_type ponita --dataloader_type ponita_nbody --trainer.learning_rate 1.8955963499765176 --model.hidden_features 128 --model.num_layers 6 --trainer.steps_per_epoch 1000 --trainer.test_macros_every 10
```

(Flags after --config overwrite config.yaml defaults)

Alternatively, feel free to make use of the provided `config.yaml` file.

See `utils/config_models.py` for other parameters and how to change them.

## Running Inference and Plotting Macros

Inference and macro plotting with its evaluation is automatic. You can tweak the frequency of this using the `--trainer.test_macros_every` flag.

If needed, you can also run this on demand:

```
python -m self_feed --config runs/ponita/2025-08-27_13-09-57/config.yaml --trainer.model_path runs/ponita/2025-08-27_13-09-57/model_best_valid_loss.pth
```
(self-feed)

```
python -m helper_scripts.visualize --folder runs/af3/2025-08-29_08-21-34/checkpoints/14/trajectories_data --sim-index=0-10
```
(macro plotting and evaluation)

## Running in the cloud (Lambda Labs)

(full API reference: https://cloud.lambdalabs.com/api/v1/docs)

### Setup

- log in to your Lambda Labs account

- create an API key in the Lambda Labs console

- set your lambda API key as an environment variable for convenience

```bash
export LAMBDA_API_KEY=YOUR-API-KEY
```

- generate a local ssh key if you don't have one

You can also use the `setup_lambda_full.sh` script to automate launching an
instance, syncing a dataset, and preparing the Docker environment. The script
accepts various command line options to override defaults such as GPU type and
dataset name. The Dockerfile is selected automatically based on the GPU type
(e.g. `Dockerfile_gh200` for GH200 instances). When the dataset name is
provided, any directories whose names start with that dataset name (for example
`DATASET_extra`) will also be copied to the remote instance. Large `*.pt` files
inside those directories are skipped to reduce bandwidth:

```bash
./setup_lambda_full.sh -t gpu_1x_a10 -d my_dataset_name
```

Run `./setup_lambda_full.sh -h` for the full list of available parameters.

```bash
ssh-keygen -t ed25519 -C "your@email.com"
```

- add your existing key to lambda

```bash
curl -u $LAMBDA_API_KEY: https://cloud.lambdalabs.com/api/v1/ssh-keys -d '{
  "name": "my-key",
  "public_key": "$(cat ~/.ssh/id_ed25519.pub)"
}' -H "Content-Type: application/json"
```

- check available instance types and pricing

```bash
curl -u $LAMBDA_API_KEY: https://cloud.lambdalabs.com/api/v1/instance-types | jq .
```

### Launching an instance and connect to it

- launch an instance

```bash
curl -u $LAMBDA_API_KEY: https://cloud.lambdalabs.com/api/v1/instance-operations/launch -d '{
"region_name": "us-east-3",
"instance_type_name": "gpu_1x_gh200",
"ssh_key_names": ["my-key"],
"file_system_names": [],
"quantity": 1
}' -H "Content-Type: application/json"
```

this prints out the instance id. Remember it.

- check the instance ip

```bash
curl -u $LAMBDA_API_KEY: https://cloud.lambdalabs.com/api/v1/instances/YOUR-INSTANCE-ID | jq .
```

- wait for it to boot, then connect and build

```bash
ssh ubuntu@YOUR-INSTANCE-IP
```

- clone the repo

- generate a new ssh key exclusive to lambda deployments

(on your local machine)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/lambda_deploy_key -N ""
```

- add the key to ssh keys on github (https://github.com/settings/keys)

- add the private key to your lambda instance:

```bash
scp ~/.ssh/lambda_deploy_key ubuntu@YOUR-INSTANCE-IP:~/.ssh/
ssh ubuntu@YOUR-INSTANCE-IP "chmod 600 ~/.ssh/lambda_deploy_key"
```

- set up Github ssh config on lambda

```bash
ssh ubuntu@YOUR-INSTANCE-IP "echo 'Host github.com
IdentityFile ~/.ssh/lambda_deploy_key' >> ~/.ssh/config"
```

(the two steps above are combined in the `helper_scripts/setup_lambda_ssh.sh` script)

- clone the repo

```bash
git clone git@github.com:Simona-Biosystems/n_body_approx.git && cd n_body_approx
```

(alternatively, if you also want to copy untracked files, rsync the repo to the instance)

### Building and running the container

```bash
# Build with host UID/GID to avoid permission issues
docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t nbody-cuda .
nvidia-smi # check gpu access
docker run --env-file .env --gpus all -it -v $(pwd):/n_body_approx nbody-cuda
```

### Copying files from the instance

- copy the files you want to keep for example using `rsync` (dry run first with --dry-run)

\# TODO: verify this works

```bash
rsync -avz --include='_/' \
 --include='_/checkpoints/_/generated_trajectories/\*\*/plots/_.json' \
 --include='_/avg_p_values_vs_checkpoints.png' \
 --include='_/individual*p_values_vs_checkpoints.png' \
 --include='*/interactive*avg_p_values_vs_checkpoints.html' \
 --exclude='*' \
 ubuntu@YOUR-INSTANCE-IP:/path/to/n_body_approx/runs/ ./local_backup/runs/
```

If you also want to persist the installed packages, you can just rsync the venv directory:

```bash
rsync -avz ubuntu@YOUR-INSTANCE-IP:/home/ubuntu/venv/ ./local_backup/venv/
```

Note: it's possible to also use Lambdalabs' filesystems to persist data between launches, but usually it's not worth it, since the instance is not guaranteed to be in the same region when you want to use it again (and the filesystem and the instance need to be in the same region since you cannot move either of them to another region. Furthermore, you can only access a filesystem from a running instance). Read more at https://docs.lambdalabs.com/public-cloud/filesystems/

### Shutting down the instance

- get instance id

```bash
curl -u $LAMBDA_API_KEY: https://cloud.lambdalabs.com/api/v1/instances | jq .
```

- terminate the instance

```bash
curl -u YOUR-API-KEY: https://cloud.lambdalabs.com/api/v1/instance-operations/terminate -d '{
"instance_ids": ["YOUR-INSTANCE-ID"]
}' -H "Content-Type: application/json"
```

- if you wanna terminate the instance automatically after some time, you can do so using something like:

```bash
(sleep 6h && ...) &
```

### Running the container on a GH200 GPU

```bash
docker build -t nbody-cuda-gh200 -f Dockerfile_gh200 .
docker run --gpus all -it -v $(pwd):/n_body_approx nbody-cuda-gh200
```
