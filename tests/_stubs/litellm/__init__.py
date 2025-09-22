drop_params = False
set_verbose = False
suppress_debug_info = False
api_base = None
api_key = None


class exceptions:
    class RateLimitError(Exception):
        pass


class BadRequestError(Exception):
    pass


class ModelResponse:
    def __init__(self, **kwargs):
        self.choices = kwargs.get("choices", [])

        usage = kwargs.get("usage", {})
        if isinstance(usage, dict):
            model_extra = usage.get("model_extra", {})
            self.usage = type("Usage", (), {"model_extra": model_extra})()
        else:
            self.usage = usage

    def json(self):
        return {}


def completion(*args, **kwargs):
    raise NotImplementedError("Stub completion should be monkeypatched in tests")
