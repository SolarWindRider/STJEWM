"""ST-JEWM evaluation entry points.

Replaces 17 old stage* eval/plan/closed_loop scripts.

Available:
    lewm_protocol  — goal-conditioned CEM planning (LeWM App. F.1 protocol)
    closed_loop    — closed-loop rollout with CEM planning + env-native success
    plan_then_render — closed-loop + GIF output
    report         — aggregate per-env JSONs into final table
"""
from .lewm_protocol import LeWMProtocol, LeWMEvalResult, lewm_evaluate, result_to_dict, save_result
from .closed_loop import (
    ClosedLoopResult, eval_closed_loop, make_env,
    parse_args as closed_loop_parse_args, main as closed_loop_main,
)

__all__ = [
    "LeWMProtocol", "LeWMEvalResult", "lewm_evaluate", "result_to_dict", "save_result",
    "ClosedLoopResult", "eval_closed_loop", "make_env",
    "closed_loop_parse_args", "closed_loop_main",
]
