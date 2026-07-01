# Provenance: Dual RTX 4090 Workstation

Role in study: machine #2 in the hardware strategy. GPU reference and large-N
ceiling. Establishes CPU-vs-GPU speedup, tests whether the second GPU helps a
single forward pass, and is the only machine that can realistically approach the
150,000-sample upper claim. Reached by `ssh ubuntu`.

## Hardware (verified)
Hostname: FPT-server
CPU: AMD Ryzen Threadripper PRO 5955WX, 16 cores / 32 threads
System memory: 128 GB (free reports 125 GiB usable)
Architecture: x86_64
GPU: 2x NVIDIA GeForce RTX 4090, 24564 MiB each
NVIDIA driver: 580.159.03

## OS and toolchain (verified)
OS: Ubuntu 22.04.5 LTS
System Python: 3.10.12
nvcc: not on PATH (no system CUDA toolkit)

## Environment notes
System Python 3.10.12 is below TabFM's hard floor of Python >= 3.11. A newer
Python (3.12, provisioned via uv) is required before installing TabFM. Install
path is `pip install -e .[jax,cuda]` (jax[cuda12]); the CUDA libraries ship in the
jax cuda12 pip wheels, so a system CUDA toolkit / nvcc is not required to run.

Verified relevant architecture fact: the JAX backend auto-builds a device mesh
over jax.devices() and shards the 32-member ensemble across all visible GPUs for a
single predict call. So both 4090s are exercised automatically on the JAX path
with no user code. The PyTorch backend has no multi-GPU code.

JAX, jaxlib-cuda, Flax, CUDA-via-wheel versions: filled in after the venv is built
and the JAX classification example runs on GPU.
