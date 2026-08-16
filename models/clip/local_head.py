import torch.nn as nn


class LocalPatchHead(nn.Module):
    """
    CLIP 局部 Patch 轻量适配头。

    输入:
        x: [B, N_patch, D] 或 [N_patch, D]

    输出:
        x': 与输入形状一致

    设计原则:
        1. 只学习局部 Patch 修正，不改变 CLIP Backbone。
        2. 输出层 zero-init，初始时严格满足 x' = x。
        3. 不在 Head 内做 L2 Normalize，Region pooling 后再统一归一化。
    """

    def __init__(self, dim=512):
        super().__init__()

        self.proj = nn.Linear(
            dim,
            dim,
        )

        # 初始时 residual correction = 0，
        # 因而 Enhanced Patch 与原始 CLIP Patch 完全一致。
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        return x + self.proj(x)
