# RunPod directional continuation

This handoff runs only the portable continuation of the already-passed
quartile-directional preflight, replay, and controls. It does not regenerate
paths, train a model, open confirmation evidence, or alter the three immutable
source runs.

## Relocation experiment note

The source child is frozen at `ready_for_fittrace` with 26 registered
artifacts. Its two parents contain the already-open historical physical-fit,
gain-calibration, and training-rank evidence needed by later stages. A local
end-to-end relocation seam verified all three trees, reconstructed the
root-independent identity
`a25b766fe6174db985e71946c0a6bcb656747ff95a1f376811defaff7f2070ed`,
replayed the GPU resource pilot, and ended `ready_for_fittrace`. No scientific
stage beyond the already-passed controls was opened by this seam.

The RunPod child is a continuation of the same frozen evidence, not a new
experiment or a relaxation of provenance. If relocation, runtime, fittrace,
nomination, or adjudication fails, the launcher stops after committing readable
evidence. It never falls back to regenerating paths or retraining models.

The original child cannot be resumed directly on a Linux pod because its
scientific configuration binds Windows absolute paths. The launcher therefore
uses the additive
`mnist.diag_d0_jacobi_rb_quartile_directional_portable_continuation` workflow,
which first content-verifies and relocates the immutable evidence into a fresh
child run.

## Pod profile

Use an on-demand RunPod Pod, not an interruptible instance. The authorizing
runtime is frozen to:

- Python `3.14.4`;
- PyTorch `2.11.0+cu128` and CUDA runtime `12.8`;
- NumPy `2.4.4`;
- CUDA compute capability `12.0` (an RTX 5090 is the recommended RunPod GPU);
- at least 8 GB GPU memory, 16 GiB host RAM, and four vCPUs;
- a 40--50 GiB persistent volume mounted at `/workspace`.

The predecessor used Torch's default nondeterministic-algorithm mode, disabled
CUDA-matmul TF32, enabled cuDNN TF32, and had no `CUBLAS_WORKSPACE_CONFIG`.
Neither setup nor launch scripts change those settings; runtime verification
fails closed if the pod differs.

RunPod Pods are Linux-only. Files under `/workspace` survive a Pod stop, but a
normal volume is deleted when the Pod is terminated. Use a network volume or
download the result before termination. See the official RunPod documentation
for [storage](https://docs.runpod.io/pods/storage/types) and
[SCP/rsync transfer](https://docs.runpod.io/pods/storage/transfer-files).

## Build and upload on Windows

From the repository root, with no local directional stage running:

```powershell
.\.venv\Scripts\python.exe tools\runpod_directional\build_bundle.py `
  --output handoff\jacobi_directional_runpod_20260808.zip

Get-FileHash handoff\jacobi_directional_runpod_20260808.zip -Algorithm SHA256
```

The builder writes a Zip64 `ZIP_STORED` archive and
`jacobi_directional_runpod_20260808.zip.sha256`. It contains source, the two
relevant documents, and exactly these immutable run trees:

- the 4,120-artifact quartile-specialist parent;
- the 29-artifact time-local parent;
- the 26-artifact directional child at `ready_for_fittrace`.

It excludes `.git`, virtual environments, temporary/test directories,
`mnist_data`, and all unrelated run history. Every payload file has an internal
SHA-256 commitment. Use full SSH with a public IP for a large SCP transfer; the
basic proxied SSH connection does not support SCP:

```powershell
scp -P <PORT> -i <KEY> `
  handoff\jacobi_directional_runpod_20260808.zip `
  handoff\jacobi_directional_runpod_20260808.zip.sha256 `
  root@<POD_IP>:/workspace/
```

## Verify and install on RunPod

```bash
cd /workspace
sha256sum -c jacobi_directional_runpod_20260808.zip.sha256
python3 -m zipfile -e jacobi_directional_runpod_20260808.zip /workspace

cd /workspace/condition_df
python3 tools/runpod_directional/verify_bundle.py --root /workspace/condition_df
bash tools/runpod_directional/install_env.sh
```

The installer places Python, its package cache, and `.venv-runpod` on persistent
storage. It installs the exact cu128 wheels and finishes by checking the bundle,
versions, GPU capability, memory, and inherited Torch backend defaults.

## Launch and monitor

Install `tmux` if the selected template does not provide it, then start the
fail-closed continuation:

```bash
apt-get update && apt-get install -y tmux
cd /workspace/condition_df
tmux new-session -d -s jacobi-directional \
  'bash /workspace/condition_df/tools/runpod_directional/run_stage.sh continue'
```

Closing the browser or SSH connection does not stop the tmux job. Monitor it
with:

```bash
tmux attach -t jacobi-directional
# Detach with Ctrl-b, then d.

tail -f /workspace/condition_df/runpod_runtime/directional-*.log
bash tools/runpod_directional/run_stage.sh status
```

`continue` creates the relocation child when absent and advances only through
`fittrace`, `nominate`, `adjudicate`, and `report` after each required gate
passes. It stops immediately on an unknown status, nonzero exit, or gate
failure. Candidate artifacts are restartable: after a Pod restart, run the same
tmux command again and committed work is skipped.

Before terminating the Pod, download the complete new directory under:

```text
runs/experiment12_d0_jacobi_rb_quartile_directional_portable_continuation/
```
