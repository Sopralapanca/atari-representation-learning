"""Central gymnasium import and ALE registration.

Use `from atariari.gym import gym` across the codebase so ALE is registered once.
"""
try:
    import gymnasium as gym
except Exception:  # fallback if gymnasium isn't available
    gym = None

try:
    # Register ALE environments when available
    if gym is not None:
        import ale_py
        try:
            gym.register_envs(ale_py)
        except Exception:
            # some versions may not need explicit registration
            pass
except Exception:
    # ale_py not installed or registration failed; leave gym as-is
    pass

__all__ = ["gym"]
