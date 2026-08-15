import torch.nn as nn


class ResidualAdapter(nn.Module):
    """轻量残差 Adapter：只学习特征修正量 Δx。"""

    def __init__(self, dim=512, bottleneck_dim=128):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)

        # 初始修正量为 0，保证加入 Adapter 后起点等于原 CLIP。
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(x)))