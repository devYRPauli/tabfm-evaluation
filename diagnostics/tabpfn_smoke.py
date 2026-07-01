#!/usr/bin/env python3
"""Isolated TabPFN smoke test with an internal alarm (macOS has no `timeout`).
Tells us whether TabPFN authenticates + downloads weights and runs, or hangs."""
import os
import signal

print("token in python env:", "set" if os.environ.get("TABPFN_TOKEN") else "MISSING")
import numpy as np
from sklearn.datasets import make_classification
from tabpfn import TabPFNClassifier


class Timeout(Exception):
    pass


def _handler(signum, frame):
    raise Timeout()


signal.signal(signal.SIGALRM, _handler)
signal.alarm(100)

X, y = make_classification(n_samples=100, n_features=5, random_state=0)
try:
    c = TabPFNClassifier()
    print("fitting (100s alarm)...")
    c.fit(X[:80], y[:80])
    print("proba0:", c.predict_proba(X[80:81]).round(3).tolist())
    print("TABPFN_OK")
except Timeout:
    print("TABPFN_TIMEOUT: fit did not finish in 100s (weight-download network hang)")
except Exception as exc:
    print("TABPFN_ERROR:", repr(exc)[:400])
