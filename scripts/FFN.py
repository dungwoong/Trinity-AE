import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "frontend"))

import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, M, N, N4, WO=None, WFF1a=None, WFF1b=None, WFF2=None, device=None, dtype=None):
        super().__init__()
        self.M = M
        self.N = N
        self.N4 = N4
        self.device = device
        self.dtype = dtype

        # nn.Linear layers (bias=False)
        self.fc_o = nn.Linear(N, N, bias=False)
        self.fc_ff1a = nn.Linear(N, N4, bias=False)
        self.fc_ff1b = nn.Linear(N, N4, bias=False)
        self.fc_ff2 = nn.Linear(N4, N, bias=False)

        # Initialize weights with provided values
        # nn.Linear weight shape is (out_features, in_features)
        # torch.matmul(x, W) = x @ W, but nn.Linear does x @ W.T
        # So we need to transpose the weights
        if WO is not None:
            self.fc_o.weight.data = WO.T.to(device=device, dtype=dtype)
        if WFF1a is not None:
            self.fc_ff1a.weight.data = WFF1a.T.to(device=device, dtype=dtype)
        if WFF1b is not None:
            self.fc_ff1b.weight.data = WFF1b.T.to(device=device, dtype=dtype)
        if WFF2 is not None:
            self.fc_ff2.weight.data = WFF2.T.to(device=device, dtype=dtype)

    def forward(self, O2, X):
        attn_O1 = self.fc_o(O2)
        attn_O2 = attn_O1 + X
        attn_O3 = attn_O2.pow(2).mean(-1, keepdim=True)
        attn_O_norm = attn_O2 * torch.rsqrt(attn_O3)

        FF1a = self.fc_ff1a(attn_O_norm)
        FF1b = self.fc_ff1b(attn_O_norm)
        FF1b_silu = FF1b * torch.sigmoid(FF1b)

        FF1 = FF1a * FF1b_silu
        FF2 = self.fc_ff2(FF1)

        return FF2


if __name__ == "__main__":
    import trinity

    M, N, N4 = 16, 4096, 16384

    WO = torch.randn(N, N)
    WFF1a = torch.randn(N, N4)
    WFF1b = torch.randn(N, N4)
    WFF2 = torch.randn(N4, N)

    X = torch.randn((M, N))
    O2 = torch.randn(M, N)

    model = FFN(M, N, N4, WO=WO, WFF1a=WFF1a, WFF1b=WFF1b, WFF2=WFF2)
    result = trinity.optimize(model, (O2, X), basename="ffn")
