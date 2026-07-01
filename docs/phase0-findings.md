# Phase 0 Findings: Code Reading and the Weight-Format Resolution

All claims below are read from the upstream repo
(github.com/google-research/tabfm, cloned at upstream/tabfm, HEAD 443cbec,
2026-06-30). The repo is ground truth over the blog and over the brief. File and
line references are to that clone.

## 1. The JAX-vs-pytorch question, resolved

There are two separate, independently hosted Hugging Face weight repos. They are
not the same artifact and they are not the same file format.

1. JAX, canonical and source of truth. `tabfm/src/jax/tabfm_v1_0_0.py:34` sets
   `HF_REPO_ID = "google/tabfm-1.0.0-jax"`. `load()` snapshot-downloads that repo
   and restores an Orbax checkpoint directory through `flax.nnx`
   (`tabfm/src/jax/tabfm_v1_0_0.py:206-236`, with the Orbax CheckpointManager in
   `tabfm/src/jax/checkpointing.py:22-39`). The on-disk format is an Orbax
   checkpoint directory, not a single binary file.
2. PyTorch, derived export. `tabfm/src/pytorch/tabfm_v1_0_0.py:24` sets
   `HF_REPO_ID = "google/tabfm-1.0.0-pytorch"`. `load()` downloads
   `pytorch_model.bin` and loads it with `torch.load(...)` then
   `load_state_dict(..., strict=True)` (`tabfm/src/pytorch/tabfm_v1_0_0.py:103-120`).
   The format is a vanilla pickled state_dict (a `.bin`), not safetensors,
   despite the repo name.
3. Provenance of the PyTorch weights. `tabfm/src/hugging_face/convert_and_upload.py`
   loads the JAX model, converts parameters with
   `tabfm/src/hugging_face/torch_convert.py` (unstacks vmapped blocks, transposes
   Linear kernels, remaps names), runs a parity check requiring max abs diff
   below 1e-4, then `torch.save` to `pytorch_model.bin`. So the PyTorch repo is a
   validated conversion of the JAX model, not an independently trained model.

Both repos are public and download with no token. The example scripts default to
the JAX backend; the PyTorch path is present but commented out. Google's own
published eval result files are named `tabfm-jax-tpu-tabarena-*.parquet`, meaning
Google's reference numbers were produced on JAX plus TPU.

Decision for this study: JAX is the primary backend (faithful to the canonical
artifact and the only backend with multi-GPU support). PyTorch is kept as a
secondary portability cross-check.

## 2. License is split across two artifacts

1. Code (GitHub repo, root LICENSE, `pyproject.toml:8,11`): Apache License 2.0,
   permissive.
2. Weights (both HF repos, from HF cardData): license "other", license_name
   `tabfm-non-commercial-v1.0`.

The brief's non-commercial note is correct, but it applies to the weights only.
The reproduction repo must state both, because a permissive code license and a
non-commercial weight license govern different artifacts.

## 3. The public API (verified signatures)

1. JAX load: `tabfm.tabfm_v1_0_0_jax.load(model_type="classification",
   checkpoint_path=None, step=None, *, col_attention_impl="flash",
   row_attention_impl="jax", icl_attention_impl="flash", dtype=jnp.bfloat16,
   use_cache=True)` (`tabfm/src/jax/tabfm_v1_0_0.py:127-137`). Process-wide load
   cache keyed by all settings.
2. PyTorch load: `tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification",
   checkpoint_path=None, *, device=None, use_cache=True)`
   (`tabfm/src/pytorch/tabfm_v1_0_0.py:72-79`).
3. `TabFMClassifier(model, n_estimators=32, ..., max_num_features=500,
   max_num_rows=None, softmax_temperature=0.9, ..., n_feature_crosses=0,
   n_svd_features=0, enable_nnls=False, ...)` and the same minus class-specific
   args for `TabFMRegressor` (`tabfm/src/classifier_and_regressor.py:1861-1889,
   2752`). The cross-feature, SVD, NNLS 32-way, and Platt-calibration controls
   that make up TabFM-Ensemble are constructor args, plus an `.ensemble()` preset
   classmethod. TabFM-Ensemble does not need to be reimplemented.
4. `fit(X, y)` does not train the network. Its docstring states the model uses
   in-context learning at inference time and fit only prepares the data
   transformations (`tabfm/src/classifier_and_regressor.py:2012-2013, 2867-2868`);
   the code builds encoders, scalers, and the ensemble generator, with no
   optimizer.
5. `predict_proba(X) -> [T, K]`, `predict(X) -> [T]` (argmax), regressor
   `predict(X) -> [T]`.

## 4. Hard limits and gotchas

1. max_classes = 10 (`tabfm/src/jax/tabfm_v1_0_0.py:42`). `fit` raises ValueError
   above it (`classifier_and_regressor.py:2047-2051`). Phase 3 classification
   datasets must have 10 or fewer classes. `max_num_features=500` and
   `max_num_rows` are soft, user-overridable subsampling defaults, not model
   limits.
2. Python >= 3.11 is a hard install floor (driven by flax >= 0.12.7). Verified
   environment impact: Mac Studio system Python 3.9.6 and workstation Python
   3.10.12 are both too old, so a newer Python is provisioned with uv on those
   machines.
3. Multi-GPU helps the JAX backend only. The JAX path auto-builds a device mesh
   over `jax.devices()` and shards the 32-member ensemble across all visible
   devices for a single predict call, no user code
   (`classifier_and_regressor.py:2246-2264, 3027-3043`). The PyTorch backend has
   no parallelism code, so the second 4090 only helps on JAX.
4. First-run JIT compilation takes minutes. Wall-clock benchmarks must separate
   warmup from steady-state.
5. No jax-metal reference anywhere in the repo. Apple Silicon GPU acceleration is
   not wired in; JAX runs on CPU on the Macs.

## 5. Package layout (one line each)

1. `tabfm/__init__.py`: public surface, conditional backend imports,
   `__version__ = '1.0.0'`.
2. `tabfm/src/classifier_and_regressor.py` (3321 lines): sklearn-compatible
   classifier and regressor, all preprocessing transformers, ensemble generation,
   and the JAX/PyTorch forward-pass dispatch including device sharding.
3. `tabfm/src/jax/model.py` (2779 lines): the Flax NNX TabFM architecture (column
   embedding, row interaction, ICL transformer).
4. `tabfm/src/jax/tabfm_v1_0_0.py`: JAX release config and `load()`.
5. `tabfm/src/jax/checkpointing.py`: Orbax save/restore helpers.
6. `tabfm/src/jax/memory_efficient_attention.py` (627 lines): flash and
   memory-efficient attention kernels.
7. `tabfm/src/pytorch/model.py` (453 lines): PyTorch port of the architecture.
8. `tabfm/src/pytorch/tabfm_v1_0_0.py`: PyTorch release config and `load()`.
9. `tabfm/src/hugging_face/torch_convert.py`: JAX to PyTorch weight converter and
   parity test.
10. `tabfm/src/hugging_face/convert_and_upload.py`: CLI that converts,
    parity-validates, and uploads the PyTorch weights.
