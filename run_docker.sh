#!/bin/bash
# Run Isaac-GR00T Docker image with GPU passthrough and checkpoints mounted

IMAGE="gr00t:dev"
HOST_CHECKPOINTS="$(pwd)/checkpoints"
CONTAINER_CHECKPOINTS="/workspace/checkpoints"
HOST_DEPLOY="$(pwd)/g1_deploy/"
CONTAINER_DEPLOY="/workspace/g1_deploy/"
HOST_DATASETS="$(pwd)/datasets"
CONTAINER_DATASETS="/workspace/datasets"

# Optional: X11 forwarding for GUI apps
X11_FLAGS="-v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY"

# # Run the container
# exec docker run --gpus all -it --rm \
#   --shm-size=8g \
#   # --network macvlan_net --ip 192.168.123.90 \
#   --network host \
#   $X11_FLAGS \
#   -v "$HOST_CHECKPOINTS":"$CONTAINER_CHECKPOINTS" \
#   -v "$HOST_DEPLOY":"$CONTAINER_DEPLOY" \
#   "$IMAGE" "$@"

# Run the container
exec docker run --gpus all -it --rm \
  --shm-size=8g \
  --network macvlan \
  --ip 192.168.123.71 \
  $X11_FLAGS \
  -v "$HOST_CHECKPOINTS":"$CONTAINER_CHECKPOINTS" \
  -v "$HOST_DEPLOY":"$CONTAINER_DEPLOY" \
  -v "$HOST_DATASETS":"$CONTAINER_DATASETS" \
  "$IMAGE" "$@"
