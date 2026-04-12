"""
PyTorch MiniGPT training benchmark — measures real-world ML overhead.
Runs a small transformer training loop (forward + backward + optimizer).
Returns (wall_ms, gpu_ms) for the timed region.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

# ── Model ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(C, dim=2)
        nh = self.n_head
        hs = C // nh
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size=512, block_size=128, n_embd=768, n_head=12, n_layer=6):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.head(self.ln_f(self.blocks(x)))
        return x

# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_training(steps: int = 20, warmup_steps: int = 5, batch_size: int = 16,
                 seq_len: int = 128, use_scope: bool = False) -> tuple[float, float]:
    """Run MiniGPT training, return (wall_ms, gpu_ms)."""
    device = torch.device('cuda')
    model = MiniGPT().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler()

    # Warm-up
    for _ in range(warmup_steps):
        x = torch.randint(0, 512, (batch_size, seq_len), device=device)
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Optionally import gpufl for scoped timing
    scope_ctx = None
    if use_scope:
        try:
            import gpufl
            scope_ctx = gpufl.Scope
        except ImportError:
            pass

    # Timed region
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    wall_start = time.perf_counter()
    start_event.record()

    for step in range(steps):
        x = torch.randint(0, 512, (batch_size, seq_len), device=device)

        if scope_ctx:
            with scope_ctx(f"step_{step}"):
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    end_event.record()
    torch.cuda.synchronize()
    wall_end = time.perf_counter()

    gpu_ms = start_event.elapsed_time(end_event)
    wall_ms = (wall_end - wall_start) * 1000.0

    # Cleanup
    del model, optimizer, scaler
    torch.cuda.empty_cache()

    return wall_ms, gpu_ms


if __name__ == '__main__':
    wall, gpu = run_training()
    print(f"MiniGPT 20 steps: wall={wall:.1f}ms, gpu={gpu:.1f}ms")
