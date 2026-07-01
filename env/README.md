# Environments and Locks

TabFM requires Python >= 3.11. We provision Python 3.12 with uv on every machine
and install the JAX backend (primary). PyTorch is a secondary cross-check path
installed separately where used.

## Why we do not use Google's requirements.txt directly

Google's `requirements.txt` is a single pip-compiled lock that bundles both the
jax and pytorch extras and pins `torch==2.12.1+cpu`. That `+cpu` torch wheel
exists only for Linux and Windows, never macOS arm64. So the lock is categorically
unsatisfiable on Apple Silicon, and the install never even reaches the jaxlib pin.
On the Macs we install the `.[jax]` extras route, which has no torch dependency.

Divergence from Google's lock that results: jax and jaxlib resolve to 0.10.2
rather than the locked 0.10.1 (one patch newer, latest available). flax lands
exactly on the locked 0.12.7. This divergence is recorded so cross-run differences
are attributable.

## Our locks (pinned for reproducibility)

We pin our own resolved versions so all machines and future re-runs use the same
stack, rather than floating to whatever is latest at install time.

1. `lock-macos-arm64-cpu.txt`: full freeze of the Mac Studio venv (Python 3.12.13,
   JAX CPU). Used verbatim on both Macs (Studio and MacBook), same arch. jax 0.10.2.
2. `lock-linux-x86_64-cuda.txt`: to be captured from the workstation venv (Python
   3.12, jax[cuda12]). Same arch only on the workstation. Shared pure-Python deps
   pinned to match the Mac lock where the resolver allows; jaxlib is the cuda12
   variant rather than the CPU variant. Version parity (jax 0.10.2, flax 0.12.7)
   is verified and any unavoidable difference is documented.

Cross-platform note: exact byte-for-byte parity across arm64-mac and x86_64-linux
is not possible (different jaxlib build, different transitive wheels). We hold the
jax and flax versions equal across machines and document the rest, which keeps the
CPU-vs-GPU timing comparison clean at the level that matters.
