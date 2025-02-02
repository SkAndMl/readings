import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional

class GPTConfig:
    max_seq_len: int = 256
    embed_dim: int = 256
    ff_multiply: int = 4
    n_head: int = 4
    n_blocks: int = 6
    vocab_size: int = 1000

class FFN(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(config.embed_dim, config.embed_dim*config.ff_multiply),
            nn.ReLU(),
            nn.Linear(config.embed_dim*config.ff_multiply, config.embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class Attn(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.QKV = nn.Linear(config.embed_dim, 3*config.embed_dim)
        self.O = nn.Linear(config.embed_dim, config.embed_dim)
    
    def forward(self, x: torch.Tensor, kv_cache: Optional[List[torch.Tensor]]=None):
        b, s, d = x.shape
        qkv: torch.Tensor = self.QKV(x)
        q, k, v = qkv.split(split_size=d, dim=-1)
        q = q.view((b, s, self.n_head, d//self.n_head)).transpose(1, 2)
        k = k.view((b, s, self.n_head, d//self.n_head)).transpose(1, 2)
        v = v.view((b, s, self.n_head, d//self.n_head)).transpose(1, 2)

        if kv_cache is not None:
            old_k, old_v = kv_cache
            k = torch.cat((old_k, k), dim=2)
            v = torch.cat((old_v, v), dim=2)
            mask = None
        else:
            mask = torch.triu(
                input=torch.ones((b, 1, s, s)),
                diagonal=1
            )* (-1e10)
        
        wts = (q @ k.transpose(-2, -1)) / (d//self.n_head)**(0.5)
        if mask is not None: wts += mask
        wts = F.softmax(wts, dim=-1) # b, n_head, s, s
        y = wts @ v # b, n_head, s, head_dim
        y = y.contiguous().transpose(1, 2).view((b, s, d))
        
        return self.O(y), ([k, v] if kv_cache is not None else None)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ffn = FFN(config)
        self.attn = Attn(config)
    
    def forward(self, x: torch.Tensor, kv_cache: Optional[List[torch.Tensor]]=None):
        attn_x, kv_cache = self.attn(x, kv_cache)
        x += attn_x
        x = self.ffn(x) + x
        return x, kv_cache

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.embed_dim//self.n_head
        self.gpt = nn.ModuleDict({
            "wte": nn.Embedding(num_embeddings=config.vocab_size, embedding_dim=config.embed_dim),
            "wpe": nn.Embedding(num_embeddings=config.max_seq_len, embedding_dim=config.embed_dim),
            "blocks": nn.ModuleList([Block(config) for _ in range(config.n_blocks)]),
            "lm_head": nn.Linear(config.embed_dim, config.vocab_size)
        })
    
    def forward(self, x: torch.Tensor, kv_caches=None):
        b, s = x.shape
        wte = self.gpt.wte(x)
        wpe = self.gpt.wpe(torch.arange(0, s, device=x.device).unsqueeze(0))
        x = wte + wpe.unsqueeze(0)
        for i, block in enumerate(self.gpt.blocks):
            if kv_caches: x, kv_caches[i] = block(x, kv_caches[i])
            else: x, _ = block(x)
        
        logits = self.gpt.lm_head(x)
        return logits, kv_caches
        
    def generate(self, input_ids: torch.Tensor, max_tokens: int=10, use_kv_cache: bool=False):
        b, s = input_ids.shape
        x = input_ids.clone()
        kv_caches = None
        if use_kv_cache:
            kv_caches = {
                i:torch.empty((b, self.n_head, s, self.head_dim))
                for i in range(len(self.gpt.blocks))
            }
        # run 1 iter to setup kv_cache
        logits, kv_caches = self(x, kv_caches)
        next_tokens = torch.argmax(logits[:, -1:, :], dim=-1)
        x = torch.cat([x, next_tokens], dim=1)

        for _ in range(max_tokens-1):
            if use_kv_cache:
                logits, kv_caches = self(x[:, -1:], kv_caches)
                next_tokens = torch.argmax(logits[:, -1:, :], dim=-1)
                x = torch.cat([x, next_tokens], dim=1)
            else:
                logits, _ = self(x[:, -1:], None)
                next_tokens = torch.argmax(logits[:, -1:, :], dim=-1)
                x = torch.cat([x, next_tokens], dim=1) 
        
        return x