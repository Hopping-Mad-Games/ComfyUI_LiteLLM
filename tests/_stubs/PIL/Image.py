from io import BytesIO


class _StubImage:
    def __init__(self, data=None):
        self.data = data or []

    def convert(self, mode):
        return self

    def save(self, buffer: BytesIO, format=None, quality=None):
        buffer.write(b"")


def fromarray(arr):
    return _StubImage(arr)


def open(*args, **kwargs):
    return _StubImage()
