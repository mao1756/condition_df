# Unattended RunPod experiment

This workflow runs the complete one-image Jacobi/Rao--Blackwell capacity study:

1. build fresh K=512 eager-prefix training and validation caches;
2. train the existing 34,974-parameter controller with the old loss;
3. train the same controller with the path-weighted loss;
4. train the 2,390,174-parameter controller with the path-weighted loss;
5. perform paired same-forward-endpoint and independent-Dirichlet-start rollouts;
6. render reports and contact sheets, verify and archive the run; and
7. delete the Pod only after durable results have been checksum-verified.

Use a RunPod PyTorch development image with CUDA, `nvcc`, at least 24 GB GPU
memory, and an attached network volume mounted at `/workspace`. The explicit
`RUNPOD_RESULTS_DURABLE=1` assertion is required before the launcher will permit
Pod deletion.

```bash
cd /workspace/condition_df
RUNPOD_RESULTS_DURABLE=1 \
RESULTS_ROOT=/workspace/results/jacobi-path-weighted-capacity \
RUNPOD_FINAL_ACTION=delete \
bash tools/runpod_weighted_e2e/launch.sh
```

The command returns immediately after detaching the worker. Progress is written
to the printed log path. Completed caches and training checkpoints are reused if
the worker is relaunched with the same `RUN_DIR`.

For verified S3 export, set `RESULTS_URI=s3://bucket/prefix` and provide AWS
credentials through RunPod secrets. Without a verified durable archive, the
finalizer stops rather than deletes the Pod.

Useful overrides are `MNIST_INDEX`, `TRAIN_PATHS`, `VALIDATION_PATHS`,
`SMALL_UPDATES`, `LARGE_UPDATES`, `BATCH_SIZE`, `VALIDATION_INTERVAL`,
`MOBILITY_FLOOR`, and `HARD_WALL_SECONDS`. The frozen defaults are 64 training
paths, 32 validation paths, 12,000 updates per architecture, batch size 32,
`MOBILITY_FLOOR=1e-4`, and a six-hour hard wall with an independent fifteen-minute
stop watchdog.
