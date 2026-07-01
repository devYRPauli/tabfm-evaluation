# Findings: Multi-GPU Sharding and Commit Pinning

## 1. Multi-GPU auto-sharding is broken via the public predict path

Verified on the workstation (Ubuntu 22.04, 2x RTX 4090, driver 580.159.03,
jax[cuda12]==0.10.2). JAX enumerates both GPUs correctly:
`jax.devices() -> [CudaDevice(id=0), CudaDevice(id=1)]`, backend gpu, using the
pip-bundled CUDA12 wheels with no system CUDA toolkit. So GPU detection is fine.

But running the stock example scripts with both GPUs visible crashes every time
through the public `TabFMClassifier` / `TabFMRegressor` predict path. Two distinct
bugs were reproduced:

1. Default `batch_size=1` plus two visible GPUs. `num_data_shards` is computed as
   1 from `jax.sharding.get_mesh()`, but on the first compiled call the code
   rebuilds `data_sharding` over all `jax.devices()` (both GPUs) without
   recomputing the shard count to match, so a batch of size 1 is forced into a
   2-way shard: `IndivisibleError: array axis 0 is partitioned 2 times, but the
   dimension size is 1`.
2. Workaround `batch_size=32` (ensemble evenly divisible by 2). Different failure:
   the model weights are placed on GPU0 only and never replicated or sharded,
   while the input is sharded across both GPUs, giving
   `ValueError: Received incompatible devices ... device ids [0] ... device ids
   [0, 1]`.

This directly answers a Phase 4 question from the brief. The blog and the code
structure imply the 32-member ensemble auto-shards across visible devices, so a
second GPU would help a single forward pass. In practice, at the tested commit,
a second visible GPU does not help and actively breaks the public predict path.
True multi-GPU data-parallel inference would require explicit, non-default
mesh and sharding setup that the bundled examples do not exercise.

Validated working GPU configuration: pin a single GPU with
`CUDA_VISIBLE_DEVICES=0`. Both examples then pass with real allocation (about
22.7 GiB on GPU0 for the 32-member ensemble), GPU1 idle. For throughput, two
independent single-GPU jobs can run in parallel, one per card.

Consequence for Phase 4: the GPU datapoint is single-4090. The "does the second
4090 help a single forward pass" experiment has a verified answer (no, it
currently breaks), which we report as a finding rather than a speedup number. If
time allows we can test whether an explicit user-provided mesh makes dual-GPU work,
but that is beyond reproducing the shipped behavior.

## 2. Commit drift across machines, now pinned

The upstream repo is under active commits on release day. The clones landed on
two different HEADs:

1. Mac Studio: `443cbec`. Phase 1 and Phase 2 harness were developed and validated
   against this commit.
2. MacBook and workstation: `b6ea70b` ("Merge pull request #26 from
   google-research/fix-ci-skip-backend-tests").

The diff between the two commits was inspected. It touches only
`.github/workflows/pytest_and_autopublish.yml` and `conftest.py`, both CI and
test-config. There are zero changes under `tabfm/src`, so the model and inference
code is byte-identical between the two commits. The intervening commits are
`5c01e06` (skip backend test modules when an optional extra is absent) and
`b12b1ec` (CI installs both extras), again test-only.

Decision: pin every machine to `b6ea70b`. It is the newer commit and two of three
machines are already on it. Because `tabfm/src` is identical, the Studio's Phase 1
and Phase 2 results obtained at 443cbec remain valid at b6ea70b with no re-run
needed. The editable install reflects the checked-out source, so the pin is a
`git checkout b6ea70b` in `~/tabfm-eval/upstream/tabfm` on the Studio (applied
after its in-flight Phase 2 run completes, to avoid changing source under a running
process), with no reinstall. All resolved package versions already match exactly
across machines (jax 0.10.2, flax 0.12.7, and the rest).

The multi-GPU sharding bugs in section 1 are present at b6ea70b (where they were
found) and, since the sharding code is unchanged, at 443cbec as well.
