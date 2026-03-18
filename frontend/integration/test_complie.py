import torch
from torch import nn
from integration import trinity_backend

# 1. 모델 정의
class SimpleAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.d_model = d_model
        self.heads = heads
        self.head_dim = d_model // heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch, seq, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1))
        attn = torch.softmax(scores, dim=-1)

        output = torch.matmul(attn, v)
        return output
    

# 모델 설정
d_model = 64
heads = 4
batch = 1
seq = 16

# 모델 생성
model = SimpleAttention(d_model, heads).cuda()
example_input = torch.randn(batch, seq, d_model).cuda()

print("--- 1. Original Run ---")
out_ref = model(example_input)

# 2. 컴파일 (Trinity Backend 사용)
# 첫 실행(Warmup) 시점에 trinity_backend가 호출되어 컴파일 수행
print("\n--- 2. Compiling with Trinity ---")
opt_model = torch.compile(model, backend=trinity_backend)

# # 3. 실행 (Compiled Kernel Run)
out_trinity = opt_model(example_input)

# print("\n--- 3. Verification ---")
# # 결과가 같은지 확인 (오차 범위 내)
# print(f"Match: {torch.allclose(out_ref, out_trinity, atol=1e-4)}")
