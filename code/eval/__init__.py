"""ST-JEWM evaluation entry points.

Replaces 17 old stage* eval/plan/closed_loop scripts.

Available:
    closed_loop    — closed-loop rollout with CEM planning + env-native success
    plan_then_render — closed-loop + GIF output
    report         — aggregate per-env JSONs into final table
"""
from .closed_loop import (
    ClosedLoopResult, eval_closed_loop, make_env,
    parse_args as closed_loop_parse_args, main as closed_loop_main,
)

__all__ = [
    "ClosedLoopResult", "eval_closed_loop", "make_env",
    "closed_loop_parse_args", "closed_loop_main",
]
