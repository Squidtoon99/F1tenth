__all__ = ["F1tenthEnv"]


def __getattr__(name):
    if name == "F1tenthEnv":
        from .env import F1tenthEnv

        return F1tenthEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
