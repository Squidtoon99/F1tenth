__all__ = ["F1tenthEnv"]


def __getattr__(name):
    if name == "F1tenthEnv":
        from .warp_env import WarpF1tenthEnv

        return WarpF1tenthEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
