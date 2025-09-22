"""Lightweight torch stub for unit tests."""
from __future__ import annotations

import numpy as np


class Tensor:
    def __init__(self, array):
        self._array = np.array(array)

    def mul(self, value):
        self._array = self._array * value
        return self

    def byte(self):
        self._array = self._array.astype(np.uint8)
        return self

    def numpy(self):
        return np.array(self._array)

    def float(self):
        self._array = self._array.astype(np.float32)
        return self

    def div(self, value):
        self._array = self._array / value
        return self

    def contiguous(self):
        return self


def from_numpy(array):
    return Tensor(array)


def stack(tensors, dim=0):
    arrays = [
        t.numpy() if isinstance(t, Tensor) else np.array(t)
        for t in tensors
    ]
    return Tensor(np.stack(arrays, axis=dim))

