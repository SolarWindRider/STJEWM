"""Event-Window task: agent reports the modal event of the most recent
N-step window. Tests whether the trace's content-aware α gate
outperforms the membrane readout and ALIF-timecell's passive decay.

Setup:
- 5 event types (E0, E1, E2, E3, E4).
- Each event type has a *natural rate* λ_i per step.
- Per step, exactly ONE event type is drawn (sampled with probability
  proportional to λ_i, normalised).
- Observation: 5D one-hot of the *most recently drawn event* + 5D
  rate vector (the current λ values).
- Action: 5D categorical (the agent's guess for the modal event of
  the *current window*).
- Reward: +1 if the agent's guess matches the true modal event of
  the current window at the window boundary, 0 otherwise.

The *trick*: a *switching event* flips the rate pattern at a random
window boundary (with probability `switch_prob`). If the agent's
predictor remembers the *previous* rate pattern (content-aware trace),
it can detect the switch and *ignore* the early-window events from
the old pattern. If it just decays passively (ALIF-timecell), it conflates
pre-switch and post-switch events — and the modal event in the
switch-window will look like an old-pattern event, leading to a wrong
guess.

Why this tests the trace's α gate:
- Spike on event dim i drives r_i up by 1.
- The α gate is high when the *current* input is *consistent* with the
  *current trace* — so if event i has been firing for many steps, the
  trace on dim i is high, and the gate sees "consistent", and α stays
  high. A different dim firing means the gate sees "changed", and α
  drops. The trace on the *old* dim decays selectively, while the new
  dim's trace accumulates.
- The trace therefore acts as a *content-aware rate counter that
  resets on input change*. ALIF-timecell cannot do this (its τ_k is fixed; it
  decays both dims at their own rates regardless of which one is
  currently firing).
- Membrane readout cannot do this either (v_t reflects the current
  LIF state, not the recent rate).

The agent must answer "what's the modal event of the *most recent
window*?" — a question whose answer requires *integration* of recent
input (which dim has been firing most) combined with *selective
retention* (which dim's recent firing is most relevant, given that
the rate pattern may have switched).

State machine
=============
- We treat each *step* as: env draws event e_t, increments its window
  count, then obs_{t+1} = (one_hot(e_t), rates_t). The agent's
  action_t is its guess for the *window* containing e_t (i.e. the
  window we're currently in). At the window boundary, the env scores
  the agent's *last* action against the window's modal event.
- Concretely: action_t is scored at step t+1 if t+1 is the last step
  of the current window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np

from .base import BaseEnv, EnvSpec


# Event types are integers 0..4
N_EVENTS = 5
OBS_DIM = 10  # 5D one-hot (most recent event) + 5D rates (current λ)
ACTION_DIM = 5  # categorical pick
EPISODE_LEN_STEPS = 200  # 20 windows × 10 steps
WINDOW_LEN = 10
N_WINDOWS = EPISODE_LEN_STEPS // WINDOW_LEN  # 20


# Natural event rates (sums to 1.0). Pattern A: E2 is dominant.
# Pattern B: E0 is dominant. They differ in *which* event is
# common, so the agent must track the *current* dominant event.
RATE_PATTERN_A = np.array([0.05, 0.05, 0.40, 0.30, 0.20], dtype=np.float64)
RATE_PATTERN_B = np.array([0.40, 0.20, 0.05, 0.10, 0.25], dtype=np.float64)


@dataclass
class EventWindowConfig:
    window_len: int = WINDOW_LEN
    n_windows: int = N_WINDOWS
    n_events: int = N_EVENTS
    switch_prob: float = 0.30
    seed: Optional[int] = None

    @property
    def episode_len(self) -> int:
        return self.window_len * self.n_windows


class EventWindowEnv(BaseEnv):
    ENV_KIND = "event_window"

    def __init__(self, cfg: Optional[EventWindowConfig] = None):
        super().__init__()
        self.cfg = cfg or EventWindowConfig()
        self.spec = EnvSpec(
            env_id=f"event_window/win{self.cfg.window_len}_w{self.cfg.n_windows}",
            obs_dim=OBS_DIM,
            action_dim=ACTION_DIM,
            action_low=np.full(ACTION_DIM, -1.0, dtype=np.float32),
            action_high=np.full(ACTION_DIM, 1.0, dtype=np.float32),
            obs_keys=("state",),
            max_episode_steps=self.cfg.episode_len,
        )
        self._step_count = 0
        self._window_count = 0
        self._step_in_window = 0
        self._rng = np.random.default_rng(self.cfg.seed)
        self._current_pattern = RATE_PATTERN_A.copy()
        self._last_drawn_event: int = -1  # set in reset
        # Per-window event counts. Reset at window boundary AFTER scoring.
        self._window_event_counts: Dict[int, int] = {i: 0 for i in range(self.cfg.n_events)}
        # The agent's "pending guess" for the current window. Updated
        # every step from the action. Scored at window boundary.
        self._pending_guess: int = 0
        # The modal event of the most recently completed window (for
        # the agent to read at the boundary). Computed at the boundary.
        self._last_window_modal: int = -1

    def reset(self, seed: Optional[int] = None, **kwargs) -> Dict[str, np.ndarray]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._window_count = 0
        self._step_in_window = 0
        self._current_pattern = RATE_PATTERN_A.copy()
        self._last_drawn_event = -1
        self._window_event_counts = {i: 0 for i in range(self.cfg.n_events)}
        self._pending_guess = 0
        self._last_window_modal = -1
        # Draw the *first* event so obs_0 has a non-trivial one-hot.
        return self._draw_and_observe()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """At time t, the agent has already seen obs_t = (one_hot(e_{t-1}), rates).
        The action is the agent's guess for "what is the modal event
        of the *current window* (which includes e_{t-1})?".

        We do *not* score the action at this step — we score the
        action taken *at the last step of the window* (i.e. the action
        that the agent chose when it saw the *last* event of the
        window). The reward for the action at step t is computed when
        the window boundary is reached.

        This is a non-standard MDP but it's the standard for "guess the
        modal event of the past K events" tasks.
        """
        # 1) Decode the agent's action (categorical pick)
        action = np.asarray(action, dtype=np.float32).flatten()
        if action.shape[0] >= self.cfg.n_events:
            self._pending_guess = int(np.argmax(action[:self.cfg.n_events]))
        else:
            self._pending_guess = int(np.argmax(action))

        # 2) Compute the reward for the action the agent gave at the
        # *previous* step (if we're now at the *end* of a window).
        # Specifically: at the end of a window (step_in_window == window_len-1),
        # the action we *just received* is the agent's guess for the
        # *current* window. Score it against the modal event.
        reward = 0.0
        if self._step_in_window == self.cfg.window_len - 1:
            modal_event = max(self._window_event_counts, key=self._window_event_counts.get)
            self._last_window_modal = modal_event
            reward = 1.0 if self._pending_guess == modal_event else 0.0
            # Maybe switch the rate pattern for the next window
            if self._rng.random() < self.cfg.switch_prob:
                self._current_pattern = (
                    RATE_PATTERN_B.copy() if self._current_pattern is RATE_PATTERN_A
                    else RATE_PATTERN_A.copy()
                )
            # Reset the per-window counts (they will be re-incremented
            # as the next window's events are drawn).
            self._window_event_counts = {i: 0 for i in range(self.cfg.n_events)}

        # 3) Draw the next event (for the NEXT step's obs)
        obs = self._draw_and_observe()

        # 4) Bookkeeping
        self._step_in_window += 1
        if self._step_in_window >= self.cfg.window_len:
            self._step_in_window = 0
            self._window_count += 1
        self._step_count += 1
        done = self._step_count >= self.cfg.episode_len

        info: Dict[str, Any] = {
            "step": self._step_count,
            "step_in_window": self._step_in_window,
            "window_count": self._window_count,
            "true_event": self._last_drawn_event,
            "guess": self._pending_guess,
            "current_pattern": "A" if self._current_pattern is RATE_PATTERN_A else "B",
            "last_window_modal": self._last_window_modal,
        }
        if done:
            info["final_window_count"] = self._window_count
            info["final_score"] = reward  # last reward (informational)
        return obs, reward, done, info

    def _draw_and_observe(self) -> Dict[str, np.ndarray]:
        """Draw the next event, increment window counts, build the
        obs (one-hot of the just-drawn event + current rates)."""
        e_t = int(self._rng.choice(self.cfg.n_events, p=self._current_pattern))
        self._last_drawn_event = e_t
        self._window_event_counts[e_t] += 1
        one_hot = np.zeros(self.cfg.n_events, dtype=np.float32)
        one_hot[e_t] = 1.0
        rates = self._current_pattern.astype(np.float32).copy()
        state = np.concatenate([one_hot, rates])
        return {"state": state, "obs": state}

    def get_state(self) -> np.ndarray:
        return self._obs_dict()["state"].copy()

    def _obs_dict(self) -> Dict[str, np.ndarray]:
        """Build the obs from the current state (no new draw)."""
        one_hot = np.zeros(self.cfg.n_events, dtype=np.float32)
        if self._last_drawn_event >= 0:
            one_hot[self._last_drawn_event] = 1.0
        rates = self._current_pattern.astype(np.float32).copy()
        state = np.concatenate([one_hot, rates])
        return {"state": state, "obs": state}

    def check_success(self, state: np.ndarray, goal_state: np.ndarray) -> Tuple[bool, float]:
        """No env-native success notion; the cumulative reward is the metric.
        The eval pipeline's primary metric is the reward returned by step."""
        return False, 0.0


def make_event_window(seed: Optional[int] = None) -> EventWindowEnv:
    return EventWindowEnv(EventWindowConfig(seed=seed))
