float32 = float


def zeros(shape, dtype=None):
    if isinstance(shape, int):
        shape = (shape,)

    def build(level):
        if level == len(shape) - 1:
            return [0] * shape[level]
        return [build(level + 1) for _ in range(shape[level])]

    return build(0)


def array(data, dtype=None):
    return data


def expand_dims(arr, axis):
    return [arr]


def asarray(data, dtype=None):
    return data


def float32_cast(value):
    return float(value)
