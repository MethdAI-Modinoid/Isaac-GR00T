from huggingface_hub import snapshot_download

snapshot_download(repo_id="nvidia/GR00T-N1.5-3B", local_dir="./checkpoints/GR00T-N1.5-3B")
