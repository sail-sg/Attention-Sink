"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from lit_gpt.config import Config
from xformers.ops import SwiGLU

KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")
    

def _get_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)

    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    return (
        get_slopes_power_of_2(closest_power_of_2) + _get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
    )


def get_slopes(num_heads: int, batch_size: int, device: torch.device):
    return (
        torch.Tensor(_get_slopes(num_heads))
        .unsqueeze(1)
        .unsqueeze(1)
        .unsqueeze(0)
        .repeat(batch_size, 1, 1, 1)
        .to(device)
    )


class GPTrelpos(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []

        self.attention_bias = None

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        # print module name
        if isinstance(module, nn.Embedding):
            # RWKV: set it to 1e-4
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1)))
            # torch.nn.init.normal_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            # fan-in variance scaling intializer
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / module.weight.size(1)))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX       
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (name == "w3.weight" and isinstance(module, SwiGLU)):  #if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(p.shape[-1])  /  n_layer)
        

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache is not None and self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-gpt/pull/83#issuecomment-1558150179
            # self.rope_cache = None
            self.mask_cache = None
            self.attention_bias = None
    

    def forward(
        self, idx: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = idx.size()
        use_kv_cache = input_pos is not None

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        if use_kv_cache:  # not relevant otherwise
            assert (
                max_seq_length >= T
            ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"
        
        if self.attention_bias is None:
            self.attention_bias = self.build_attention_bias(idx)

        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099
        if use_kv_cache and self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        # cos, sin = self.rope_cache
        if use_kv_cache:

            # cos = cos.index_select(0, input_pos)
            # sin = sin.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            # cos = cos[:T]
            # sin = sin[:T]
            mask = None


        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        
        if self.config.embd_pdrop > 0:
            x = nn.functional.dropout(x, p=self.config.embd_pdrop,
                                      training=self.training)
        
        if not use_kv_cache:
            for block in self.transformer.h:
                x, *_ = block(x, max_seq_length, attention_bias=self.attention_bias)
        else:
            self.kv_caches = self.kv_caches or self.build_kv_caches(x, max_seq_length, self.config.head_size)
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(x,  max_seq_length, mask, input_pos, self.kv_caches[i], attention_bias=self.attention_bias)

        x = self.transformer.ln_f(x)

        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))
    

    def build_mask_cache(self, idx: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((self.config.block_size, self.config.block_size), device=idx.device, dtype=torch.bool)
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)
    

    def build_attention_bias(self, idx: torch.Tensor) -> torch.Tensor:
        # build casual mask
        attn_bias = torch.zeros(1, self.config.n_head, self.config.block_size, self.config.block_size, dtype=torch.bfloat16, device=idx.device)
        temp_mask = torch.ones(self.config.block_size, self.config.block_size, dtype=torch.bool).tril(diagonal=0).to(idx.device)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        
        # build relative position
        position = torch.arange(self.config.block_size, device=idx.device)
        relative_position = position[None, :] - position[:, None]
        relative_position = (relative_position.unsqueeze(0).unsqueeze(0).repeat(1, self.config.n_head, 1, 1)).to(torch.bfloat16)
        # build relative positional embedding
        if self.config.positional_embedding == "alibi":
            slopes = get_slopes(num_heads=self.config.n_head, batch_size=1, device=idx.device)
            slopes = slopes.to(torch.bfloat16)
            alibi = slopes * relative_position
            return attn_bias + alibi

        elif self.config.positional_embedding == "relative":
            num_buckets = 32
            max_distance = 128
            relative_buckets = 0
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

            # half of the buckets are for exact increments in positions
            max_exact = num_buckets // 2
            is_small = relative_position < max_exact

            # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
            relative_position_if_large = max_exact + (
                torch.log(relative_position / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).to(torch.long)
            relative_position_if_large = torch.min(
                relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
            )

            relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
            return attn_bias + relative_buckets
        else:
            raise NotImplementedError
    

    def build_kv_caches(self, idx: torch.Tensor, max_seq_length: int) -> List[KVCache]:
        B = idx.size(0)
        heads = 1 if self.config.n_query_groups == 1 else self.config.n_query_groups

        k_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        v_cache_shape = (B, max_seq_length, heads, self.config.head_size)
        device = idx.device
        return [
            (torch.zeros(k_cache_shape, device=device), torch.zeros(v_cache_shape, device=device))
            for _ in range(self.config.n_layer)
        ]


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config
    def forward(
        self,
        x: torch.Tensor,
        max_seq_length: int,
        attention_bias: Optional[torch.Tensor] = None,
        # mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        n_1 = self.norm_1(x)
        h, new_kv_cache = self.attn(n_1, max_seq_length, attention_bias, input_pos, kv_cache)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            ffn = self.mlp(n_2)
            x = x + h + ffn
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )
            
            x = x + h
            ffn = self.mlp(self.norm_2(x))
            if self.config.resid_pdrop:
                ffn = nn.functional.dropout(ffn, p=self.config.resid_pdrop, training=self.training)
            x = x + ffn
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        max_seq_length: int,
        attention_bias: Optional[torch.Tensor] = None,
        # mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        from .fused_rotary_embedding import apply_rotary_emb_func

        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B,  T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.config.head_size)  
        v = v.reshape(B,  T, -1, self.config.head_size)  

        # n_elem = int(self.config.rotary_percentage * self.config.head_size)
    
        # q_roped = apply_rope(q[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # k_roped = apply_rope(k[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # print( (q_roped - q).sum())
        # q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        # k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=1)
                cache_v = torch.roll(cache_v, -1, dims=1)

            k = cache_k.index_copy_(1, input_pos, k)
            v = cache_v.index_copy_(1, input_pos, v)
            kv_cache = k, v
        
        y = self.scaled_dot_product_attention(q, k, v, attention_bias=attention_bias)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y, kv_cache

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_bias: Optional[torch.Tensor] = None,
        # mask: Optional[torch.Tensor] = None
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        
        # if (
        #     FlashAttention2Available
        #     and mask is None
        #     and q.device.type == "cuda"
        #     and q.dtype in (torch.float16, torch.bfloat16)
        # ):
        #     from flash_attn import flash_attn_func
        #     return flash_attn_func(q, k, v, dropout_p=self.config.attn_pdrop, softmax_scale=scale, causal=True)
    
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        
        # print(attention_bias)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_bias, dropout_p=self.config.attn_pdrop, scale=scale, is_causal=False,
        )
        return y.transpose(1, 2)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        # self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        # self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
        self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False, _pack_weights=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x_fc_1 = self.fc_1(x)
        # x_fc_2 = self.fc_2(x)
        # x = torch.nn.functional.silu(x_fc_1) * x_fc_2
        # return self.proj(x)
        return self.swiglu(x)
