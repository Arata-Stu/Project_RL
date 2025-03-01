import torch
import numpy as np
from collections import deque
from .base import ReplayBufferBase

class OffPolicyBuffer(ReplayBufferBase):
    def __init__(self, size: int, state_dim: int=64, action_dim: int=3, n_step: int=3, gamma: float=0.99):
        super().__init__(size, state_dim, action_dim)
        self.n_step = n_step
        self.gamma = gamma
        self.temp_buffer = deque(maxlen=n_step)
    
    def add(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool):
        self.temp_buffer.append((state, action, reward, next_state, done))
        if len(self.temp_buffer) >= self.n_step:
            self._store_n_step_transition()
        if done:
            while self.temp_buffer:
                self._store_n_step_transition()
    
    def _store_n_step_transition(self):
        state, action, _, _, _ = self.temp_buffer[0]
        reward = 0
        discount = 1
        _, _, _, next_state, done = self.temp_buffer[-1]

        for _, _, r, _, d in self.temp_buffer:
            reward += discount * r
            discount *= self.gamma
            if d:
                break

        idx = self.position
        self.buffer["state"][idx] = self._to_numpy(state)
        self.buffer["action"][idx] = self._to_numpy(action)
        self.buffer["reward"][idx] = reward
        self.buffer["next_state"][idx] = self._to_numpy(next_state)
        self.buffer["done"][idx] = done

        self.position = (self.position + 1) % self.size
        if self.position == 0:
            self.full = True

        self.temp_buffer.popleft()