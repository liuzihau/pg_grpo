# two_stage_curriculum.py
from __future__ import annotations

class Curriculum:
    """
    Linear ramp with an initial 'hold' phase to stabilize early training.
    """
    def __init__(
        self,
        total_kd_steps: int,
        max_input_len_start: int = 512,
        max_input_len_end: int = 2048,
        max_new_tokens_start: int = 128,
        max_new_tokens_end: int = 512,
        hold_ratio: float = 0.25,  # hold first 25% of steps at the 'start' sizes
    ):
        self.T = max(1, int(total_kd_steps))
        self.in_s, self.in_e = max_input_len_start, max_input_len_end
        self.new_s, self.new_e = max_new_tokens_start, max_new_tokens_end
        self.hold = int(self.T * max(0.0, min(hold_ratio, 0.9)))

    def at(self, step: int):
        t = min(max(step, 0), self.T)
        if t <= self.hold:
            return self.in_s, self.new_s
        a = (t - self.hold) / max(1, self.T - self.hold)
        max_input_len = int(round(self.in_s + a * (self.in_e - self.in_s)))
        max_new_tokens = int(round(self.new_s + a * (self.new_e - self.new_s)))
        return max_input_len, max_new_tokens
