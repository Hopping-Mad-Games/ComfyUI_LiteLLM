class BaseModel:
    @classmethod
    def schema_json(cls, *args, **kwargs):
        return "{}"

    @classmethod
    def schema(cls, *args, **kwargs):
        return {}


def Field(default=None, **kwargs):
    return default


def conlist(item_type, **kwargs):
    return list


def create_model(name, **fields):
    return type(name, (BaseModel,), fields or {})
