# Provenance: Mac Studio M4 Max

Role in study: machine #1 in the hardware strategy. Primary development machine
and the strong-CPU datapoint. JAX CPU backend. Most RAM of the two Macs. The
harness is built and stabilized here first, and large CPU contexts are pushed
here. Reached by `ssh macstudio`.

## Hardware (verified)
Model identifier: Mac16,9
CPU: Apple M4 Max
Unified memory: 64 GB (68719476736 bytes)
Architecture: arm64

## OS and toolchain (verified)
macOS: 26.5.2 (build 25F84)
System Python: 3.9.6 (/usr/bin/python3)
git: /usr/bin/git

## Environment gotcha
System Python 3.9.6 is below TabFM's hard floor of Python >= 3.11. A newer Python
(3.12, provisioned via uv) is required before installing TabFM. CPU JAX backend
only. Do not attempt jax-metal against the jax==0.10.1 pin.

JAX, Flax, and the resolved load path versions: filled in after the isolated venv
is built and the JAX classification example runs.
