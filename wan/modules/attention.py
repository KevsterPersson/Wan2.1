# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import triton
import triton.language as tl
import warnings

__all__ = [
    'flash_attention',
    'attention',
]

def cdiv(a, b):
    return (a + b - 1) // b

@triton.jit
def _simple_fwd_kernel(
    Q, K, V, sm_scale,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    # Simplified kernel for AMD compatibility
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    
    # Use simple pointer arithmetic instead of block_ptr
    q_offset = off_z * stride_qz + off_h * stride_qh
    k_offset = off_z * stride_kz + off_h * stride_kh
    v_offset = off_z * stride_vz + off_h * stride_vh
    o_offset = off_z * stride_oz + off_h * stride_oh
    
    # Initialize offsets with simple ranges
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    
    # Initialize accumulation variables
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # Load Q - use simple pointer arithmetic
    q_ptrs = Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    
    # Apply scaling
    q = (q * sm_scale).to(tl.float16)
    
    # Determine loop bounds
    lo = 0
    hi = N_CTX
    if IS_CAUSAL:
        lo = start_m * BLOCK_M
        hi = min((start_m + 1) * BLOCK_M, N_CTX)
    
    # Main computation loop
    for start_n in range(lo, hi, BLOCK_N):
        # Load K, V with simple pointers
        # Note: We need to load K in a transposed manner for the dot product
        k_ptrs = K + k_offset + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
        v_ptrs = V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        
        # Load k with shape [BLOCK_DMODEL, BLOCK_N] for proper matrix multiplication
        k = tl.load(k_ptrs, mask=(offs_n[None, :] < N_CTX) & (offs_d[:, None] < BLOCK_DMODEL), other=0.0)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX, other=0.0)
        
        # Compute QK^T: q @ k.T where q is [BLOCK_M, D] and k is [D, BLOCK_N]
        # This gives us [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k, allow_tf32=True)
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = (offs_m[:, None] >= (start_n + offs_n[None, :]))
            qk = tl.where(causal_mask, qk, float("-inf"))
        
        # Compute softmax
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        # Update accumulation
        alpha = tl.exp(m_ij - m_i)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        # acc += p @ v where p is [BLOCK_M, BLOCK_N] and v is [BLOCK_N, D]
        # This gives us [BLOCK_M, D]
        acc += tl.dot(p.to(v.dtype), v, allow_tf32=True)
        
        m_i = tl.maximum(m_i, m_ij)
    
    # Normalize and store output
    acc = acc / l_i[:, None]
    o_ptrs = Out + o_offset + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(tl.float16), mask=offs_m[:, None] < N_CTX)

def triton_simple_attention(q, k, v, causal=False, sm_scale=None):
    """
    Simplified Triton attention for AMD compatibility
    q, k, v: [batch, num_heads, seq_len, head_dim]
    """
    # Ensure tensors are contiguous and in the right format
    q = q.contiguous()
    k = k.contiguous() 
    v = v.contiguous()
    
    # Shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk <= 256  # Head dimension constraint
    
    # Use conservative block sizes for AMD
    BLOCK_M = 64
    BLOCK_N = 64
    
    if sm_scale is None:
        sm_scale = 1.0 / (Lq ** 0.5)
    
    o = torch.empty_like(q)
    
    # Grid configuration
    grid = (cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    
    # Launch kernel with conservative settings
    _simple_fwd_kernel[grid](
        q, k, v, sm_scale,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
        IS_CAUSAL=causal,
        num_warps=4,
        num_stages=1  # Reduced stages for stability
    )
    
    return o

def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # Store original dtype for output
    out_dtype = q.dtype
    
    # Convert to float16 for AMD compatibility
    def to_float16(x):
        return x.to(torch.float16) if x.dtype in (torch.float32, torch.bfloat16) else x

    # Preprocess tensors
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    
    # Handle variable length sequences
    if q_lens is None:
        q_seq = to_float16(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32, device=q.device)
    else:
        q_seq = to_float16(torch.cat([u[:v] for u, v in zip(q, q_lens)]))
    
    if k_lens is None:
        k_seq = to_float16(k.flatten(0, 1))
        v_seq = to_float16(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32, device=k.device)
    else:
        k_seq = to_float16(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v_seq = to_float16(torch.cat([u[:v] for u, v in zip(v, k_lens)]))
    
    # Apply query scaling if specified
    if q_scale is not None:
        q_seq = q_seq * q_scale
    
    # Process sequences individually to avoid complex indexing
    cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens.cumsum(0)]).to(q.device)
    cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens.cumsum(0)]).to(k.device)
    
    outputs = []
    for i in range(b):
        start_q = cu_seqlens_q[i]
        end_q = cu_seqlens_q[i+1]
        start_k = cu_seqlens_k[i]
        end_k = cu_seqlens_k[i+1]
        
        if start_q == end_q or start_k == end_k:
            # Handle empty sequences
            out_i = torch.zeros((end_q - start_q, q_seq.shape[1], q_seq.shape[2]), 
                              device=q.device, dtype=torch.float16)
            outputs.append(out_i)
            continue
            
        q_i = q_seq[start_q:end_q].unsqueeze(0)  # [1, Lq_i, H, C]
        k_i = k_seq[start_k:end_k].unsqueeze(0)  # [1, Lk_i, H, C]
        v_i = v_seq[start_k:end_k].unsqueeze(0)  # [1, Lk_i, H, C]
        
        # Convert to [batch, num_heads, seq_len, head_dim] format
        q_i = q_i.transpose(1, 2).contiguous()
        k_i = k_i.transpose(1, 2).contiguous()
        v_i = v_i.transpose(1, 2).contiguous()
        
        # Use simplified triton attention
        out_i = triton_simple_attention(
            q_i, k_i, v_i,
            causal=causal,
            sm_scale=softmax_scale,
        )
        
        # Convert back to [batch, seq_len, num_heads, head_dim] format
        out_i = out_i.transpose(1, 2).contiguous()
        outputs.append(out_i.squeeze(0))
    
    # Concatenate all outputs
    if outputs:
        x = torch.cat(outputs, dim=0)
        # Reshape back to [B, Lq, H, C] format
        x = x.unflatten(0, (b, lq))
    else:
        # Fallback for empty batch
        x = torch.zeros_like(q)
    
    return x.to(out_dtype)

def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    # Always use our simplified triton implementation for AMD
    return flash_attention(
        q=q,
        k=k,
        v=v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        window_size=window_size,
        deterministic=deterministic,
        dtype=dtype,
        version=fa_version,
    )
