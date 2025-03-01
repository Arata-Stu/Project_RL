import torch
from .base import ReplayBufferBase

class OnPolicyBuffer(ReplayBufferBase):
    def __init__(self, size: int, state_dim: int=64, action_dim: int=3):
        super().__init__(size, state_dim, action_dim)
    
    def add(self, state: torch.Tensor, action: torch.Tensor, reward: float, next_state: torch.Tensor, done: bool):
        idx = self.position
        self.buffer["state"][idx] = self._to_numpy(state)
        self.buffer["action"][idx] = self._to_numpy(action)
        self.buffer["reward"][idx] = reward
        self.buffer["next_state"][idx] = self._to_numpy(next_state)
        self.buffer["done"][idx] = done

        self.position = (self.position + 1) % self.size
        if self.position == 0:
            self.full = True
    
    def clear(self):
        """サイズがいっぱいになったら上書きしながら維持する"""
        self.position = 0
        self.full = False