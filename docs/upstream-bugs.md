# Upstream Bug Log (PR candidates)

Living log of genuine bugs in google-research/tabfm found during this study.
These are candidates for upstream PRs. Before filing any of them we re-verify the
bug ourselves against the pinned commit and check for an existing open issue or PR
that already covers it (search the symptom, look for an open PR that closes it).
Line numbers below were read during recon and are approximate until re-confirmed
at the pinned commit b6ea70b.

Pinned commit for verification: b6ea70b (tabfm/src is identical to 443cbec).

Status legend: CANDIDATE (needs dupe check and a minimal standalone repro before
filing), VERIFIED (repro reduced and dupe-checked), FILED (PR open).

---

## BUG-1: Default predict on a multi-GPU host crashes (IndivisibleError)

Status: FILED (google-research/tabfm PR #42, together with BUG-2). Verified still
present at current main; reproduced on simulated CPU devices
(XLA_FLAGS=--xla_force_host_platform_device_count=2), fixed by removing the
first-compile override that rebuilt data_sharding over all jax.devices(), with a
CI regression test. Awaiting review + CLA.
Severity: High. Any user with two or more visible GPUs hits this on the very first
`predict` / `predict_proba` call with default settings.

Summary: With more than one GPU visible and no user-set global mesh,
`TabFMClassifier` / `TabFMRegressor` default to `batch_size=1`. The forward path
computes `num_data_shards` as 1 from `jax.sharding.get_mesh()`, but on the first
compiled call it rebuilds `data_sharding` over all `jax.devices()` (every GPU)
without recomputing the shard count to match the batch. A batch of size 1 is then
forced into an N-way shard and JAX raises:
`IndivisibleError: array axis 0 is partitioned 2 times, but the dimension size is 1`.

Affected: JAX backend, hosts with >= 2 visible CUDA devices. Reproduced on
2x RTX 4090, Ubuntu 22.04, jax[cuda12]==0.10.2.

Repro: on a 2-GPU host, `python examples/classification_example.py` with both GPUs
visible. Crashes every run. Pinning `CUDA_VISIBLE_DEVICES=0` makes it pass.

Root cause (approximate): tabfm/src/classifier_and_regressor.py around lines
2246-2264 (classifier) and 3027-3043 (regressor). The mismatch is between
`num_data_shards` (derived from the active mesh, 1 here) and the rebuilt
`data_sharding` (derived from all `jax.devices()`).

Proposed fix direction: when building `data_sharding`, derive the device set from
the same source as `num_data_shards`, or guard the sharded path so a batch smaller
than the device count is replicated rather than partitioned. Needs confirmation by
reading the exact lines at b6ea70b.

Next step: reduce to a minimal standalone repro (load model, tiny fit, predict on a
2-device host), confirm at b6ea70b, then dupe-check upstream issues and PRs.

---

## BUG-2: Multi-GPU with batch_size=32 fails on weight placement (device mismatch)

Status: FILED (google-research/tabfm PR #42, same fix as BUG-1: the removed
override caused both the batch=1 indivisibility and the params-on-device-0 vs
inputs-on-all-devices mismatch)
Severity: High, and it blocks the natural workaround for BUG-1. Likely the same
root subsystem.

Summary: Setting `batch_size=32` (the ensemble size, evenly divisible by 2 GPUs)
gets past the indivisibility error but then fails because the model weights are
placed on GPU0 only while the input batch is sharded across all visible GPUs. JAX
raises `ValueError: Received incompatible devices ... model.states ... device ids
[0] ... argument X ... device ids [0, 1]`. The weights are never replicated or
sharded to match the data sharding.

Affected: JAX backend, >= 2 visible CUDA devices, `batch_size` set to engage the
sharded path. Reproduced on 2x RTX 4090.

Repro: on a 2-GPU host, construct `TabFMClassifier(model=..., batch_size=32)`, fit,
predict with both GPUs visible.

Root cause (approximate): same sharding setup in
tabfm/src/classifier_and_regressor.py. The input is wrapped with a data sharding
over all devices, but the model parameters are not placed under a matching
sharding / replication, so the jitted call sees inputs on [0,1] and params on [0].

Proposed fix direction: replicate the model parameters across the mesh (or shard
them consistently) before the sharded forward call, so params and inputs live on
the same device set. Confirm at b6ea70b.

Next step: minimal repro, confirm at b6ea70b, dupe-check. Decide whether BUG-1 and
BUG-2 are one PR (fix the multi-device path end to end) or two.

Consequence for our study: the advertised auto-shard-across-visible-GPUs behavior
does not work through the public predict path at this commit. Our GPU datapoint is
single-4090 (CUDA_VISIBLE_DEVICES=0), which works correctly.

---

## BUG-3: predict returns a dtype=object array, breaking sklearn metrics

Status: ALREADY FIXED UPSTREAM (commit 2efc01b, "Fix TabFMClassifier.predict()
returning object-dtype labels"), between the pinned b6ea70b and current main. The
fix is the exact astype(self.classes_.dtype) cast proposed here. Not filed (moot)
Severity: Low to Medium. The class advertises a scikit-learn compatible API, but
`predict` output is rejected by standard sklearn metrics.

Summary: `TabFMClassifier.predict` returns a numpy array with `dtype=object`
(values are correct Python ints). `sklearn.utils.multiclass.type_of_target`
classifies an object array as "unknown", so `accuracy_score`, `f1_score`, and other
metrics raise `ValueError: Classification metrics can't handle a mix of binary and
unknown targets` when given the raw output. A sklearn-compatible classifier should
return predictions in the dtype of `classes_`.

Affected: both backends (observed on JAX CPU). Confirmed directly: predict on a
binary task returned `dtype=object`, `shape (n,)`, values `[1, 1, 0, 0, 0]`,
`type_of_target -> unknown`, while `classes_` is int64.

Repro: fit `TabFMClassifier` on any integer-labeled task, then
`from sklearn.metrics import accuracy_score; accuracy_score(y_true, clf.predict(X))`.
Raises. Workaround is to cast: `np.asarray(clf.predict(X)).astype(classes_dtype)`.

Root cause: the predict path builds the returned label array as object dtype rather
than casting back to `classes_.dtype`. Exact location to be pinpointed in
tabfm/src/classifier_and_regressor.py predict.

Proposed fix: cast the returned labels to `self.classes_.dtype` (one line). This
restores sklearn metric interoperability with no behavior change.

Next step: locate the exact construction line at b6ea70b, write a one-line fix and
a regression test (predict output dtype equals classes_ dtype; accuracy_score
accepts it), dupe-check, then file. This is the cleanest, lowest-risk PR of the
three.

---

## Not bugs (recorded so we do not re-investigate)

1. The pip-compiled `requirements.txt` is unsatisfiable on macOS arm64 because it
   pins `torch==2.12.1+cpu` (Linux/Windows only). This is a lockfile-portability
   limitation, arguably worth a README note upstream, but it is not a code defect.
   Tracked in env/README.md.
2. (Corrected, promoted to BUG-4 candidate.) An earlier note here claimed
   row-permutation differences were about 1e-2 bfloat16 noise. That was wrong. It
   came from a broken sanity probe whose predictions were near-flat. After fixing
   the probe, permuting context row order (labels permuted consistently, identical
   data) changes predictions by up to 0.4999 max abs proba on a learnable task
   (Studio, default 32-member ensemble). See BUG-4.

---

## BUG-4: predict_proba not invariant to context row order (rare uniform collapse)

Status: CHARACTERIZED. Low severity in aggregate. Would need model-internal
instrumentation to pin the exact bf16 operation before filing, and may be judged a
numerical-precision limitation rather than a logic bug.

What is true (Studio, default 32-member ensemble, seed 0, linsep task):
1. The model is architecturally permutation-invariant over context rows. Code trace:
   at `max_num_rows=None` no index-based row selection happens (every ensemble member
   sees identical rows), and the cross-row ICL transformer runs `use_rope=False` with
   an order-agnostic boolean train/test mask. Plain attention over unmasked keys is a
   symmetric (permutation-invariant) function.
2. Same context order twice is bit-exact (max diff 0.0). The pipeline is deterministic.
3. Permuting context row order (X and y permuted together, identical data) changes
   predict_proba by only about 0.01 to 0.02 for essentially every point. This is bf16
   non-associativity under reordering.
4. Intermittently (1 of 6 random permutations tested, plus the original phase2 one),
   exactly one test point collapses to exactly [0.5, 0.5] (uniform), even when it is a
   confident, far-from-boundary point (worst case |X.w|=4.16 vs median 1.43, base
   proba [1e-4, 0.9999]). No NaN anywhere. This single point produced the 0.4999 the
   phase2 sanity layer flagged.

Why this does not threaten the benchmark: a single point among 100 shifting barely
moves accuracy, ROC AUC, or log loss. It matters only for code relying on exact
per-row probabilities or on strict row-order invariance.

Mechanism: bf16 numerical non-associativity. In exact arithmetic the prediction is
invariant; in bf16 the reduction order changes, usually by about 1e-2, and rarely
drives one point's two class logits to exactly equal values, giving a uniform
softmax. The exact op producing the equal-logit collapse was not pinned to a line in
this pass (needs instrumenting the JAX model internals).

If filed: proposed direction is to run the final ICL logit reduction in float32, or
detect and handle degenerate equal-logit outputs. Standalone repros:
diagnostics/bug4_reconcile.py (worst-point inspection) and diagnostics/bug4_frequency.py
(frequency across permutations). Dupe-check first; may land as a numerical caveat.
