# Provenance: MacBook Pro (M1 Pro)

Role in study: machine #3 in the hardware strategy. Accessibility datapoint and
story hook. The same machine used for the TurboQuant evaluation. Runs the frozen
benchmark suite last, after the harness stops changing. Lowest-RAM machine, so it
also marks the practical small-machine ceiling.

## Hardware (verified)
Model identifier: MacBookPro18,3
CPU: Apple M1 Pro
Unified memory: 16 GB (17179869184 bytes)
Architecture: arm64

## OS and toolchain (verified)
macOS: 26.5.1 (build 25F80)
Python: 3.12.12 (pyenv shim at ~/.pyenv/shims/python3)
git: /opt/homebrew/bin/git

## Notes
TabFM pins Python >= 3.11, so 3.12.12 satisfies the floor. JAX backend on this
machine is CPU only. Per the brief, do not attempt jax-metal against the hard
jax==0.10.1 pin.

JAX, Flax, CUDA, and driver versions: to be filled in after the isolated venv is
built and `tabfm_v1_0_0.load()` succeeds on this machine.

Captured automatically at session start. Other two machines (Mac Studio M4 Max,
dual-4090 workstation) get their own provenance notes when their environments are
brought up.
