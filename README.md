# ffs

`ffs` is a minimal stereo policy scaffold:

- frozen Fast-FoundationStereo backbone
- configurable stereo history length and stereo camera pairs
- proprioceptive state history as action-head condition
- configurable action horizon and action dimension
- single-node DDP training on LeRobot v3 datasets

## Shapes

```python
left:   [B, T, V, 3, H, W]   # RGB, 0..255
right:  [B, T, V, 3, H, W]   # RGB, 0..255
state:  [B, T, state_dim]
action: [B, action_horizon, action_dim]
```

`T`, `V`, `state_dim`, `action_horizon`, and `action_dim` are set by
`configs/default.yaml`. The current dataset stores both `observation.state`
and the raw parquet `action` as 16D dual-arm EEF poses:

```text
left:  x y z qw qx qy qz gripper
right: x y z qw qx qy qz gripper
```

The loader returns `state` as the absolute EEF state history. It converts the
returned `action` chunk to EEF poses relative to the last state in the history
window: positions are expressed in that EEF frame, quaternions are
`inv(q_base) * q_target`, and gripper targets remain absolute values.

`V` is the number of configured stereo pairs:

```yaml
dataset:
  root: /data/jsy/Fast-FoundationStereo/lerobot_dataset
  image_size: [256, 320]
  camera_pairs:
    - [observation.images.head_camera_left, observation.images.head_camera_right]
    - [observation.images.right_camera_left, observation.images.right_camera_right]
```

The LeRobot parquet files provide `observation.state`, `action`, and frame
indices. The loader expects the new dataset format where `action` has the same
shape and feature names as `observation.state`. RGB frames are read from:

```text
videos/{video_key}/chunk-*/file-*.mp4
```

The dataset videos are AV1 encoded, so the loader uses `imageio` with the
system `ffmpeg` executable. Frames are resized to `dataset.image_size` before
being passed to FoundationStereo.

## Action heads

The action head is selected by `head.type` in `configs/default.yaml`.
Available types are:

- `mlp`: flatten history tokens and regress the action chunk directly.
- `rdt`: RDT-style flow-matching denoising head inspired by
  `/data/jsy/open-p2p/rdt`.

Example RDT config:

```yaml
backbone:
  feature_names: [feat_04, feat_08, feat_16, feat_32]

head:
  type: rdt
  condition_token_dim: 256
  feature_queries_per_scale: 4
  disp_queries: 8
  spatial_query_num_heads: 8
  rdt:
    hidden_size: 256
    depth: 4
    num_heads: 8
    num_inference_steps: 10
```

## Smoke test

```bash
cd /data/jsy/ffs
PYTHONPATH=. python examples/smoke_test.py
```

To run with the real frozen FoundationStereo checkpoint:

```bash
cd /data/jsy/ffs
PYTHONPATH=. python examples/smoke_test.py --real-backbone
```

The default checkpoint path points to:

```text
/data/jsy/Fast-FoundationStereo/weights/20-30-48/model_best_bp2_serialize.pth
```

## Training

The first training path supports the `mlp` action head with plain action MSE:

```bash
cd /data/jsy/ffs
python scripts/train.py --config configs/default.yaml
```

For single-node multi-GPU training, use `torchrun`. `--nproc_per_node`
should match the number of GPUs you want to use:

```bash
cd /data/jsy/ffs
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/default.yaml
```

To choose specific GPUs, set `CUDA_VISIBLE_DEVICES` before `torchrun`:

```bash
cd /data/jsy/ffs
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/default.yaml
```

Notes for DDP training:

- `train.batch_size` is the per-GPU batch size, so the global batch size is
  `train.batch_size * --nproc_per_node`.
- Each process uses one GPU via `LOCAL_RANK`; the code initializes NCCL when
  `WORLD_SIZE > 1`.
- Rank 0 writes logs and checkpoints, while all ranks participate in training.

Training writes rank-0 logs to:

```text
outputs/default/metrics.jsonl
```

Each line contains `epoch`, `step`, `global_step`, `loss`, and `lr`.
Checkpoints are saved by rank 0:

- `latest.pt`: overwritten whenever a training checkpoint or final checkpoint is saved.
- `latest.yaml`: matching model config written beside `latest.pt`, ready to use
  as the evaluation config.
- `final.pt`: always written at the end of training.
- `step_{global_step}.pt`: training checkpoint written every
  `train.save_ckpt_interval` when set.

Resume from config:

```yaml
train:
  resume_from: outputs/default/latest.pt
```

Or from the command line:

```bash
cd /data/jsy/ffs
python scripts/train.py --config configs/default.yaml --resume-from outputs/default/latest.pt
```

For a short smoke run:

```bash
cd /data/jsy/ffs
python scripts/train.py --config configs/default.yaml --max-steps 1
```

## Offline evaluation

Evaluate a checkpoint on the same `demo_clean` LeRobot dataset used by the
default config:

```bash
cd /data/jsy/ffs
python scripts/eval_offline.py \
  --config outputs/rdt_version_1_clean/latest.yaml \
  --checkpoint outputs/rdt_version_1_clean/latest.pt \
  --sample-init zeros
```

For a quick wiring check, limit it to one batch:

```bash
cd /data/jsy/ffs
python scripts/eval_offline.py --max-batches 1 --batch-size 1 --num-workers 0 --sample-init zeros --no-amp
```

The script reports action-chunk `mse`, `rmse`, and `mae` over the configured
dataset. Use `--dataset-root /path/to/lerobot_dataset_clean` to override the
dataset path without editing the config.

To save query-token attention heatmaps as a video during offline evaluation:

```bash
python scripts/eval_offline.py --max-batches 1 --batch-size 1 --num-workers 0 --visualize-query-attention
```
