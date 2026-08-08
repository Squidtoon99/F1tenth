"""Internal N-agent Warp simulator modules (not a public package API)."""

__all__ = ["WarpNAgentSimulator", "build_n_agent_simulator"]


def __getattr__(name: str):
    if name in __all__:
        from gigaflow_f1tenth.sim import runtime

        return getattr(runtime, name)
    raise AttributeError(name)
