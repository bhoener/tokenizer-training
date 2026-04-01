import os
import numpy as np
import torch


class DataLoader:
    def __init__(
        self, data_dir: str, B: int, T: int, device: torch.device | None = None
    ):
        self.data_dir = data_dir
        self.B = B
        self.T = T
        self.device = device

        self.shards = os.listdir(self.data_dir)
        self.current_shard = 0

        self.__next_shard()

    def __next_shard(self) -> None:
        self.__load_next_shard()
        self.current_shard += 1
        self.idx = 0

    def __load_next_shard(self) -> None:
        assert 0 <= self.idx < len(self.shards)
        self.data = torch.tensor(
            np.fromfile(self.data_dir + self.shards[self.idx], dtype=">i4")
        )
        if self.device is not None:
            self.data = self.data.to(self.device)

    def __call__(self) -> tuple[torch.Tensor]:
        if self.idx + self.B * self.T + 1 >= len(self.data):
            self.__next_shard()

        buf = self.data[self.idx : self.idx + self.B * self.T + 1]

        xbuf = buf[:-1].view(self.B, self.T)
        ybuf = buf[1:].view(self.B, self.T)

        return xbuf, ybuf
