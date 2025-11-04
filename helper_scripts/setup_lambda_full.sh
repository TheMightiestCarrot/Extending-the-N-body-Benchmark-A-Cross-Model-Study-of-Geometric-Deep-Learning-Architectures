#!/bin/bash

# --- Default Configuration ---
# Values here can be overridden by environment variables or command line options
TARGET_INSTANCE_TYPE=${TARGET_INSTANCE_TYPE:-"gpu_1x_gh200"} # Instance type to launch
SSH_KEY_NAME=${SSH_KEY_NAME:-"patrik-server"}       # SSH key registered in Lambda Labs
LOCAL_GITHUB_SSH_KEY_PATH=${LOCAL_GITHUB_SSH_KEY_PATH:-"$HOME/.ssh/lambda_deploy_key"}
GIT_REPO=${GIT_REPO:-"git@github.com:Simona-Biosystems/Extending-the-N-body-Benchmark-A-Cross-Model-Study-of-Geometric-Deep-Learning-Architectures.git"}
GIT_BRANCH=${GIT_BRANCH:-"main"}
REPO_BASE_DIR=${REPO_BASE_DIR:-"Extending-the-N-body-Benchmark-A-Cross-Model-Study-of-Geometric-Deep-Learning-Architectures"}
DATASET_NAME=

# Paths for file operations (relative to $HOME on local, ~ on remote)
# These are computed after parsing command line options
LOCAL_CONFIG_INI_PATH=""
LOCAL_RSYNC_SRC_DIR_REL=""
REMOTE_REPO_BASE_PATH=""
REMOTE_CONFIG_INI_PATH=""
REMOTE_RSYNC_DEST_DIR=""

# Docker configuration (executed on remote instance)
# Dockerfile name is determined automatically based on instance type
DOCKER_IMAGE_TAG=${DOCKER_IMAGE_TAG:-"nbody_approx:latest"}

# Tmux configuration
TMUX_SESSION_NAME=${TMUX_SESSION_NAME:-"nbody"}
TMUX_COMMAND=${TMUX_COMMAND:-""}  # Command to run in the tmux session (optional)
# --- End Configuration ---

# --- Option Parsing ---
usage() {
    cat <<EOF
Usage: $0 [options]
  -t TYPE      Instance type to launch (default: $TARGET_INSTANCE_TYPE)
  -d NAME      Dataset name to sync (default: $DATASET_NAME)
  -b BRANCH    Git branch to checkout (default: $GIT_BRANCH)
  -i IMAGE     Docker image tag (default: $DOCKER_IMAGE_TAG)
  -s SESSION   Tmux session name (default: $TMUX_SESSION_NAME)
  -k KEY       SSH key name registered with Lambda Labs (default: $SSH_KEY_NAME)
  -r REPO      Git repository to clone (default: $GIT_REPO)
  -c CMD       Command to run in the tmux session. If the session exists, a new
               window will be created to run this command; otherwise, the session
               will start with this command. Example:
               -c "docker run --gpus all -it --rm nbody_approx:latest bash"
  -h           Display this help message and exit

Environment variables with the same names as the options can also be used.
EOF
}

while getopts ":t:d:b:i:s:k:r:c:h" opt; do
  case $opt in
    t) TARGET_INSTANCE_TYPE="$OPTARG";;
    d) DATASET_NAME="$OPTARG";;
    b) GIT_BRANCH="$OPTARG";;
    i) DOCKER_IMAGE_TAG="$OPTARG";;
    s) TMUX_SESSION_NAME="$OPTARG";;
    k) SSH_KEY_NAME="$OPTARG";;
    r) GIT_REPO="$OPTARG";;
    c) TMUX_COMMAND="$OPTARG";;
    h) usage; exit 0;;
    *) usage; exit 1;;
  esac
done
shift $((OPTIND-1))

# Recompute derived paths in case options changed values
LOCAL_CONFIG_INI_PATH="${REPO_BASE_DIR}/config.ini"
LOCAL_RSYNC_SRC_DIR_REL="${REPO_BASE_DIR}/datasets/MD/dataset_generated/${DATASET_NAME}"
REMOTE_REPO_BASE_PATH="/home/ubuntu/${REPO_BASE_DIR}"
REMOTE_CONFIG_INI_PATH="${REMOTE_REPO_BASE_PATH}/config.ini"
REMOTE_RSYNC_DEST_DIR="${REMOTE_REPO_BASE_PATH}/datasets/MD/dataset/${DATASET_NAME}"

DOCKERFILE_NAME="Dockerfile"

# --- Polling Configuration ---
POLL_INTERVAL_SECONDS=15 # How often to check instance status (seconds)
MAX_POLLS=40             # Max attempts (~10 minutes with 15s interval)
# --- End Polling Configuration ---

# --- Availability Retry Configuration ---
# If capacity is unavailable, retry until it is (Ctrl-C to abort).
AVAIL_RETRY_INTERVAL_SECONDS=${AVAIL_RETRY_INTERVAL_SECONDS:-60}
# --- End Availability Retry Configuration ---

# --- SSH Configuration ---
# Add ServerAlive options to keep connection open during long transfers
SSH_OPTIONS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10 -o ServerAliveInterval=60 -o ServerAliveCountMax=3"
# --- End SSH Configuration ---

# --- Script Logic ---
set -e # Exit immediately if a command exits with a non-zero status.
# set -o pipefail # Uncomment if you want pipelines to fail on the first error

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check prerequisites
echo "Checking prerequisites..."
if [[ -z "$LAMBDA_API_KEY" ]]; then
  echo "Error: LAMBDA_API_KEY environment variable is not set." >&2
  exit 1
fi
if ! command_exists jq; then echo "Error: jq command not found." >&2; exit 1; fi
if ! command_exists rsync; then echo "Error: rsync command not found." >&2; exit 1; fi
if ! command_exists ssh; then echo "Error: ssh command not found." >&2; exit 1; fi
if ! command_exists scp; then echo "Error: scp command not found." >&2; exit 1; fi
if [ ! -f "$LOCAL_GITHUB_SSH_KEY_PATH" ]; then
    echo "Error: Required GitHub SSH key not found at $LOCAL_GITHUB_SSH_KEY_PATH" >&2
    echo "Please generate it first using something like:" >&2
    echo "ssh-keygen -t ed25519 -f \"$LOCAL_GITHUB_SSH_KEY_PATH\" -N \"\"" >&2
    echo "(Ensure the corresponding public key is added to your GitHub account)" >&2
    exit 1
fi
echo "Local GitHub SSH key found at $LOCAL_GITHUB_SSH_KEY_PATH."

# Derive full local path now that $HOME is definitely set
LOCAL_RSYNC_SRC_DIR="$HOME/$LOCAL_RSYNC_SRC_DIR_REL"

# Add checks for optional source files/dirs
COPY_CONFIG_INI=false
if [ -f "$HOME/$LOCAL_CONFIG_INI_PATH" ]; then
    COPY_CONFIG_INI=true
    echo "Local config file found at $HOME/$LOCAL_CONFIG_INI_PATH. Will be copied."
else
    echo "Warning: Local config file $HOME/$LOCAL_CONFIG_INI_PATH not found. Skipping copy."
fi

SYNC_DATASET=false
DATASET_DIRS=()
DATASET_PARENT_DIR="$(dirname "$LOCAL_RSYNC_SRC_DIR")"
for candidate in "$LOCAL_RSYNC_SRC_DIR" "${LOCAL_RSYNC_SRC_DIR}_"*; do
    if [ -d "$candidate" ]; then
        DATASET_DIRS+=("$candidate")
    fi
done
if [ ${#DATASET_DIRS[@]} -gt 0 ]; then
    SYNC_DATASET=true
    echo "Dataset directories to sync: ${DATASET_DIRS[*]}"
else
    echo "Warning: No dataset directories found matching ${LOCAL_RSYNC_SRC_DIR##*/}*"
fi

# 2. Check availability with auto-retry until capacity appears
echo "Checking availability for instance type: ${TARGET_INSTANCE_TYPE}..."
REGION_NAME=""
while true; do
  INSTANCE_TYPES_JSON=$(curl -sS -u "${LAMBDA_API_KEY}:" "https://cloud.lambdalabs.com/api/v1/instance-types")
  if [[ $? -ne 0 || -z "$INSTANCE_TYPES_JSON" ]]; then
      echo "Warning: failed to fetch instance types. retrying in ${AVAIL_RETRY_INTERVAL_SECONDS}s..." >&2
      sleep "$AVAIL_RETRY_INTERVAL_SECONDS" || true
      continue
  fi
  if echo "$INSTANCE_TYPES_JSON" | jq -e --arg INST_TYPE "$TARGET_INSTANCE_TYPE" '
        .data | has($INST_TYPE) and (.[$INST_TYPE].regions_with_capacity_available | length > 0)
     ' > /dev/null; then
    echo "Instance type '${TARGET_INSTANCE_TYPE}' found with available capacity."
    REGION_NAME=$(echo "$INSTANCE_TYPES_JSON" | \
                  jq -r --arg INST_TYPE "$TARGET_INSTANCE_TYPE" '
                     .data[$INST_TYPE].regions_with_capacity_available[0].name
                  ')
    if [[ -n "$REGION_NAME" && "$REGION_NAME" != "null" ]]; then
      break
    else
      echo "Warning: capacity reported but region name missing; retrying in ${AVAIL_RETRY_INTERVAL_SECONDS}s..." >&2
    fi
  else
    echo "No capacity rn. retrying in ${AVAIL_RETRY_INTERVAL_SECONDS}s..."
  fi
  sleep "$AVAIL_RETRY_INTERVAL_SECONDS" || true
done
echo "Selected region: ${REGION_NAME}"

  # 4. Construct the JSON payload for launching
  LAUNCH_PAYLOAD=$(printf '{
    "region_name": "%s",
    "instance_type_name": "%s",
    "ssh_key_names": ["%s"],
    "file_system_names": [],
    "quantity": 1
  }' "$REGION_NAME" "$TARGET_INSTANCE_TYPE" "$SSH_KEY_NAME")

  echo "--- Launch Payload ---"
  echo "$LAUNCH_PAYLOAD"
  echo "----------------------"

  # 5. Call the launch API endpoint
  echo "Sending launch request..."
  LAUNCH_RESPONSE=$(curl -sS -X POST \
       -u "${LAMBDA_API_KEY}:" \
       "https://cloud.lambdalabs.com/api/v1/instance-operations/launch" \
       -d "$LAUNCH_PAYLOAD" \
       -H "Content-Type: application/json")

   if [[ $? -ne 0 ]]; then
      echo "Error: curl command failed while attempting to launch the instance." >&2
      exit 1
   fi

  echo "--- Launch Response ---"
  # Check if response is valid JSON before piping to jq
  if echo "$LAUNCH_RESPONSE" | jq -e . > /dev/null 2>&1; then
      echo "$LAUNCH_RESPONSE" | jq .
  else
      echo "Received non-JSON response from launch API:"
      echo "$LAUNCH_RESPONSE"
  fi
  echo "-----------------------"

  # 6. Parse INSTANCE_ID
  INSTANCE_ID=$(echo "$LAUNCH_RESPONSE" | jq -r '.data.instance_ids[0]')
  if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "null" ]]; then
    echo "Error: Could not parse instance ID from the launch response." >&2
    echo "Raw response was: $LAUNCH_RESPONSE" >&2
    exit 1
  fi
  echo "Successfully initiated launch. Instance ID: ${INSTANCE_ID}"

  # 7. Poll for instance IP address
  echo "Waiting for instance ${INSTANCE_ID} to become active and get an IP address..."
  INSTANCE_IP=""
  POLL_COUNT=0
  while [[ -z "$INSTANCE_IP" && "$POLL_COUNT" -lt "$MAX_POLLS" ]]; do
      echo "Polling attempt $((POLL_COUNT + 1))/${MAX_POLLS}..." # Display 1-based count

      # Execute curl and capture the exit code
      INSTANCE_DETAILS=$(curl -sS -f -u "${LAMBDA_API_KEY}:" "https://cloud.lambdalabs.com/api/v1/instances/${INSTANCE_ID}")
      CURL_EXIT_CODE=$?
      # echo "DEBUG: curl exit code: $CURL_EXIT_CODE" # Optional debug

      if [[ $CURL_EXIT_CODE -ne 0 ]]; then
          if [[ $CURL_EXIT_CODE -eq 22 ]]; then # HTTP error with -f
             echo "Warning: Received HTTP error polling instance ${INSTANCE_ID}. Exit code: ${CURL_EXIT_CODE}. Retrying..." >&2
          else # Other curl error (network, DNS, etc.)
             echo "Warning: curl command failed during polling attempt $((POLL_COUNT + 1)). Exit code: ${CURL_EXIT_CODE}. Retrying..." >&2
          fi
          sleep $POLL_INTERVAL_SECONDS
          POLL_COUNT=$((POLL_COUNT + 1)) # Use safer increment
          continue # Skip rest of the loop iteration
      fi

      # If curl succeeded, proceed to parse
      # echo "DEBUG: curl successful, parsing details..." # Optional debug
      INSTANCE_STATUS=$(echo "$INSTANCE_DETAILS" | jq -r '.data.status')
      CURRENT_IP=$(echo "$INSTANCE_DETAILS" | jq -r '.data.ip')
      JQ_EXIT_CODE=$? # Check jq's exit code

      if [[ $JQ_EXIT_CODE -ne 0 ]]; then
          echo "Warning: Failed to parse instance details JSON with jq during poll $((POLL_COUNT + 1)). Exit code: ${JQ_EXIT_CODE}. Retrying..." >&2
          # echo "DEBUG: Received data: $INSTANCE_DETAILS" >&2 # Optional debug
          sleep $POLL_INTERVAL_SECONDS
          POLL_COUNT=$((POLL_COUNT + 1)) # Use safer increment
          continue
      fi

      echo "  Status: ${INSTANCE_STATUS:-'N/A'}, IP: ${CURRENT_IP:-'N/A'}"

      # Check if instance is active and has IP
      if [[ "$INSTANCE_STATUS" == "active" && -n "$CURRENT_IP" && "$CURRENT_IP" != "null" ]]; then
          INSTANCE_IP=$CURRENT_IP # Assign the found IP to break the loop
          echo "Instance is active with IP: ${INSTANCE_IP}"
          break # Exit the while loop
      elif [[ "$INSTANCE_STATUS" == "terminated" || "$INSTANCE_STATUS" == "error" ]]; then
          echo "Error: Instance entered status '${INSTANCE_STATUS}'. Aborting." >&2
          echo "Instance details:" >&2
          echo "$INSTANCE_DETAILS" | jq . >&2
          exit 1 # Exit the script
      fi

      # Increment POLL_COUNT at the end of a successful loop iteration
      POLL_COUNT=$((POLL_COUNT + 1)) # Use safer increment

      # Sleep only if we haven't reached the max polls yet AND haven't found the IP
      if [[ $POLL_COUNT -lt $MAX_POLLS ]]; then
          # echo "DEBUG: Sleeping for $POLL_INTERVAL_SECONDS seconds..." # Optional debug
          sleep "$POLL_INTERVAL_SECONDS" || true # Ignore sleep errors
      fi
  done
  # --- End of main polling loop ---

  # Check if loop timed out
  if [[ -z "$INSTANCE_IP" ]]; then
      echo "Error: Timed out waiting for instance ${INSTANCE_ID} to become active and get an IP address after ${POLL_COUNT} attempts." >&2
      exit 1
  fi

  # 8. Instance Setup via SSH
  echo "Starting setup on instance ${INSTANCE_ID} at ${INSTANCE_IP}..."
  REMOTE_USER="ubuntu"
  SSH_TARGET="${REMOTE_USER}@${INSTANCE_IP}"
  # SSH_OPTIONS defined in config section

  echo "Waiting a bit for SSH daemon..."
  sleep 10

  # SSH connection retry loop
  MAX_SSH_RETRIES=6
  SSH_RETRY_COUNT=0
  while [[ $SSH_RETRY_COUNT -lt $MAX_SSH_RETRIES ]]; do
      SSH_RETRY_COUNT=$((SSH_RETRY_COUNT + 1)) # Use safer increment
      echo "Attempting SSH connection (${SSH_RETRY_COUNT}/${MAX_SSH_RETRIES})..."
      if ssh $SSH_OPTIONS "$SSH_TARGET" "echo SSH connection successful"; then
          echo "SSH connection established."
          break # Exit retry loop on success
      fi
      if [[ $SSH_RETRY_COUNT -lt $MAX_SSH_RETRIES ]]; then
          echo "SSH connection failed, retrying in 15 seconds..."
          sleep 15
      else
          echo "Error: Could not establish SSH connection to ${SSH_TARGET} after ${MAX_SSH_RETRIES} attempts." >&2
          exit 1 # Exit script if SSH fails after retries
      fi
  done

  # Copy GitHub Key
  echo "Copying GitHub SSH key..."
  if ! scp $SSH_OPTIONS "$LOCAL_GITHUB_SSH_KEY_PATH" "${SSH_TARGET}:~/.ssh/github_deploy_key"; then
      echo "Error: Failed to copy GitHub SSH key via scp." >&2
      exit 1
  fi

  # Configure SSH on remote
  echo "Configuring SSH on remote instance..."
  SSH_CONFIG_COMMAND="
      mkdir -p ~/.ssh && \\
      chmod 700 ~/.ssh && \\
      chmod 600 ~/.ssh/github_deploy_key && \\
      {
          echo 'Host github.com'
          echo '  HostName github.com'
          echo '  User git'
          echo '  IdentityFile ~/.ssh/github_deploy_key'
          echo '  StrictHostKeyChecking no' # Less strict for automation
      } > ~/.ssh/config && \\
      chmod 600 ~/.ssh/config && \\
      echo 'Configured ~/.ssh/config.' && \\
      # Add github.com key to known_hosts only once
      if ! grep -q 'github.com' ~/.ssh/known_hosts 2>/dev/null; then
          ssh-keyscan github.com >> ~/.ssh/known_hosts && echo 'Added github.com to known_hosts.';
      else
          echo 'github.com already in known_hosts.';
      fi
  "
  if ! ssh $SSH_OPTIONS "$SSH_TARGET" "$SSH_CONFIG_COMMAND"; then
      echo "Error: Failed during remote SSH configuration." >&2
      exit 1
  fi

  # Propagate WANDB/HPO env to remote shell for convenience (safe quoting)
  if [[ -n "$WANDB_API_KEY" ]]; then
    echo "Configuring WANDB/HPO environment on remote..."
  if ! ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export WANDB_API_KEY=$WANDB_API_KEY >> ~/.bashrc'"; then
      echo "Warning: Failed to set WANDB_API_KEY on remote." >&2
    fi
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export WANDB_MODE=${WANDB_MODE:-online} >> ~/.bashrc'" || true
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export WANDB_PROJECT=${WANDB_PROJECT:-hpo} >> ~/.bashrc'" || true
    if [[ -n "$WANDB_ENTITY" ]]; then
      ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export WANDB_ENTITY=$WANDB_ENTITY >> ~/.bashrc'" || true
    fi
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-hpo/\$USER} >> ~/.bashrc'" || true
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export HPO_PROGRESS=${HPO_PROGRESS:-bar} >> ~/.bashrc'" || true
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export HPO_EVAL_LAST_K=${HPO_EVAL_LAST_K:-3} >> ~/.bashrc'" || true
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export HPO_EVAL_MODE=${HPO_EVAL_MODE:-best_last_k} >> ~/.bashrc'" || true
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export HPO_TEST_MACROS_EVERY=${HPO_TEST_MACROS_EVERY:-5} >> ~/.bashrc'" || true
    # also export to ~/.profile so login shells (bash -l) can see them immediately
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export WANDB_API_KEY=$WANDB_API_KEY >> ~/.profile'" || true
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export WANDB_MODE=${WANDB_MODE:-online} >> ~/.profile'" || true
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export WANDB_PROJECT=${WANDB_PROJECT:-hpo} >> ~/.profile'" || true
    if [[ -n "$WANDB_ENTITY" ]]; then
      ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export WANDB_ENTITY=$WANDB_ENTITY >> ~/.profile'" || true
    fi
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-hpo/\$USER} >> ~/.profile'" || true
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export HPO_PROGRESS=${HPO_PROGRESS:-bar} >> ~/.profile'" || true
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export HPO_EVAL_LAST_K=${HPO_EVAL_LAST_K:-3} >> ~/.profile'" || true
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export HPO_EVAL_MODE=${HPO_EVAL_MODE:-best_last_k} >> ~/.profile'" || true
    ssh $SSH_OPTIONS "$SSH_TARGET" "bash -lc 'echo export HPO_TEST_MACROS_EVERY=${HPO_TEST_MACROS_EVERY:-5} >> ~/.profile'" || true
  else
    echo "Warning: WANDB_API_KEY not found locally; remote will default to offline logging."
  fi

  # Add to docker group
  echo "Setting up permissions (docker group)..."
  # Don't exit if this fails, just warn
  ssh $SSH_OPTIONS "$SSH_TARGET" "sudo usermod -aG docker ${REMOTE_USER}" || echo "Warning: Failed to add user to docker group (maybe already added or docker not installed)."

  # Git clone and checkout
  echo "Cloning repository '${GIT_REPO}' into ${REMOTE_REPO_BASE_PATH}..."
  # Use bash -c on remote to handle cd and subsequent commands reliably
  GIT_COMMANDS="cd /home/${REMOTE_USER} && git clone \"${GIT_REPO}\" \"${REPO_BASE_DIR}\" && cd \"${REMOTE_REPO_BASE_PATH}\" && git checkout \"${GIT_BRANCH}\""
  if ! ssh $SSH_OPTIONS "$SSH_TARGET" "bash -c '${GIT_COMMANDS}'"; then
      echo "Error: Failed to clone repository or checkout branch on remote instance." >&2
      # Decide if this is fatal
      # exit 1
  else
      echo "Repository cloned and branch '${GIT_BRANCH}' checked out successfully."
  fi

  # 9. Create remote dirs, copy config, rsync dataset
  echo "Preparing remote directories for file transfer..."
  REMOTE_CONFIG_DIR=$(dirname "${REMOTE_CONFIG_INI_PATH}")
  REMOTE_RSYNC_PARENT_DIR=$(dirname "${REMOTE_RSYNC_DEST_DIR}")
  # Use a single ssh command to create all necessary parent directories robustly
  if ! ssh $SSH_OPTIONS "$SSH_TARGET" "mkdir -pv \"${REMOTE_CONFIG_DIR}\" && mkdir -pv \"${REMOTE_RSYNC_PARENT_DIR}\" && echo Dirs prepared."; then
      echo "Error: Failed to create or ensure remote directories exist." >&2
      exit 1
  fi
  echo "Remote directories prepared."

  # Copy config.ini if requested
  if [ "$COPY_CONFIG_INI" = true ]; then
      echo "Copying config.ini..."
      if ! scp $SSH_OPTIONS "$HOME/$LOCAL_CONFIG_INI_PATH" "${SSH_TARGET}:${REMOTE_CONFIG_INI_PATH}"; then
         echo "Error: Failed to copy config.ini via scp." >&2
         exit 1
      fi
      echo "config.ini copied."
  fi

  # Rsync dataset directories if requested
  if [ "$SYNC_DATASET" = true ]; then
      echo "Syncing dataset directories..."
      for LOCAL_DIR in "${DATASET_DIRS[@]}"; do
          DIR_NAME=$(basename "$LOCAL_DIR")
          REMOTE_DEST="${REMOTE_RSYNC_PARENT_DIR}/${DIR_NAME}"
          echo "  -> $DIR_NAME"
          if ! rsync -av --exclude='*.pt' --progress -e "ssh $SSH_OPTIONS" "$LOCAL_DIR/" "${SSH_TARGET}:${REMOTE_DEST}/"; then
              echo "Error: rsync command failed for $DIR_NAME." >&2
              exit 1
          fi
      done
      echo "Rsync complete."
  fi

  # 10. ensure tmux is installed on remote
  echo "Ensuring tmux is installed on remote..."
  TMUX_INSTALL_COMMAND="
      set -e
      if command -v tmux >/dev/null 2>&1; then
          echo 'tmux already installed.'
      else
          echo 'tmux not found, installing via apt...'
          sudo apt-get update && sudo apt-get install -y tmux
      fi
  "
  if ! ssh $SSH_OPTIONS "$SSH_TARGET" "$TMUX_INSTALL_COMMAND"; then
      echo "Warning: Failed to ensure tmux installed on remote." >&2
      # Don't exit for this, it's non-critical
  fi

  # 11. Build Docker image on remote instance
  echo "Building Docker image '${DOCKER_IMAGE_TAG}' using '${DOCKERFILE_NAME}' on remote..."
  # Ensure cd happens before docker build using bash -c
  DOCKER_BUILD_COMMAND="cd \"${REMOTE_REPO_BASE_PATH}\" && docker build -f \"${DOCKERFILE_NAME}\" -t \"${DOCKER_IMAGE_TAG}\" ."
  if ! ssh $SSH_OPTIONS "$SSH_TARGET" "bash -c '${DOCKER_BUILD_COMMAND}'"; then
      echo "Error: Docker build failed on remote instance." >&2
      exit 1
  fi
  echo "Docker image built successfully."

  # 12. Final Summary and Connect to Tmux
  echo "---------------------------------------------"
  echo "Setup appears complete!"
  echo "Instance ID: ${INSTANCE_ID}"
  echo "Instance IP: ${INSTANCE_IP}"
  echo "---------------------------------------------"
  echo ""
  echo "attempting to connect to tmux session '${TMUX_SESSION_NAME}' on remote instance..."
  if [ -n "$TMUX_COMMAND" ]; then
      echo "a command was provided and will be executed in the tmux session:"
      echo "  $TMUX_COMMAND"
  fi
  echo "if session doesn't exist, it will be created."
  echo "hint: tmux detach is CTRL-b then d (unless remapped)."
  echo ""

  RUN_CMD_FLAG=0
  if [ -n "$TMUX_COMMAND" ]; then
      RUN_CMD_FLAG=1
      PY_INNER="${TMUX_COMMAND} ; exec bash"
      printf -v SQ_PY_INNER %q "$PY_INNER"
      DOCKER_RUN_CMD="docker run --gpus all -it --rm \
        -e WANDB_API_KEY=${WANDB_API_KEY:-} -e WANDB_MODE=${WANDB_MODE:-online} -e WANDB_PROJECT=${WANDB_PROJECT:-hpo} -e WANDB_ENTITY=${WANDB_ENTITY:-} -e WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-hpo/\$USER} \
        -e HPO_PROGRESS=${HPO_PROGRESS:-bar} -e HPO_EVAL_LAST_K=${HPO_EVAL_LAST_K:-3} -e HPO_EVAL_MODE=${HPO_EVAL_MODE:-best_last_k} -e HPO_TEST_MACROS_EVERY=${HPO_TEST_MACROS_EVERY:-5} \
        -v \"${REMOTE_REPO_BASE_PATH}\":/n_body_approx -w /n_body_approx \"${DOCKER_IMAGE_TAG}\" bash -lc ${SQ_PY_INNER}"
  fi

  TMUX_CONNECT_COMMAND="
     SESSION=\"${TMUX_SESSION_NAME}\";
     WORK_DIR=\"${REMOTE_REPO_BASE_PATH}\";
    if ! command -v tmux >/dev/null 2>&1; then
        echo 'tmux unavailable; dropping you into a shell.';
        cd \"$WORK_DIR\" && exec bash;
    fi;
    if tmux has-session -t \"$SESSION\" 2>/dev/null; then
        echo 'attaching to existing tmux session';
    else
        echo 'creating new tmux session';
        tmux new-session -d -s \"$SESSION\" -c \"$WORK_DIR\";
    fi;
    if [ \"$RUN_CMD_FLAG\" = \"1\" ]; then
        tmux new-window -t \"$SESSION\" -n run -c \"$WORK_DIR\" \"${DOCKER_RUN_CMD}\";
    fi;
    tmux attach -t \"$SESSION\";
  "

  if ! ssh -t $SSH_OPTIONS "$SSH_TARGET" "$TMUX_CONNECT_COMMAND"; then
      echo "error: failed to connect to remote tmux session." >&2
      echo "try manually:" >&2
      if [ "$RUN_CMD_FLAG" -eq 1 ]; then
          echo "1) ssh -t ${SSH_TARGET}" >&2
          echo "2) inside, run: tmux new -As ${TMUX_SESSION_NAME} -c ${REMOTE_REPO_BASE_PATH}" >&2
          echo "   then inside tmux: ${DOCKER_RUN_CMD}" >&2
      else
          echo "ssh -t ${SSH_TARGET} 'tmux new -As ${TMUX_SESSION_NAME} -c ${REMOTE_REPO_BASE_PATH}'" >&2
      fi
      exit 1
  fi
 