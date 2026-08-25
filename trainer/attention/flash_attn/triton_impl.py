import math
import functools
import torch
import triton
import triton.language as tl

# ============================================================
# FlashAttention Forward
# ============================================================

@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale,
    D: tl.constexpr, Bq: tl.constexpr, Bk: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index, batch_index = tl.program_id(0), tl.program_id(1)
    q_start = query_tile_index * Bq

    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb, shape=(N_QUERIES, D), strides=(stride_qq, stride_qd), offsets=(q_start, 0), block_shape=(Bq, D), order=(1, 0))
    O_block_ptr = tl.make_block_ptr(O_ptr + batch_index * stride_ob, shape=(N_QUERIES, D), strides=(stride_oq, stride_od), offsets=(q_start, 0), block_shape=(Bq, D), order=(1, 0))
    L_block_ptr = tl.make_block_ptr(L_ptr + batch_index * stride_lb, shape=(N_QUERIES,), strides=(stride_lq,), offsets=(q_start,), block_shape=(Bq,), order=(0,))
    K_block_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb, shape=(N_KEYS, D), strides=(stride_kk, stride_kd), offsets=(0, 0), block_shape=(Bk, D), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb, shape=(N_KEYS, D), strides=(stride_vk, stride_vd), offsets=(0, 0), block_shape=(Bk, D), order=(1, 0))

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    m_i, l_i, O_i = tl.full((Bq,), value=float("-inf"), dtype=tl.float32), tl.zeros((Bq,), dtype=tl.float32), tl.zeros((Bq, D), dtype=tl.float32)
    num_k_tiles = tl.cdiv(q_start + Bq, Bk) if is_causal else tl.cdiv(N_KEYS, Bk)

    for j in range(num_k_tiles):
        k_start = j * Bk
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        if is_causal:
            causal_mask = (q_start + tl.arange(0, Bq))[:, None] >= (k_start + tl.arange(0, Bk))[None, :]
            Sij = tl.where(causal_mask, Sij, float("-inf"))
        m_ij, m_new = tl.max(Sij, axis=1), tl.maximum(m_i, m_ij)
        alpha, Pij = tl.exp(m_i - m_new), tl.exp(Sij - m_new[:, None])
        l_i = alpha * l_i + tl.sum(Pij, axis=1)
        O_i = O_i * alpha[:, None] + tl.dot(Pij.to(Vj.dtype), Vj)
        m_i = m_new
        K_block_ptr, V_block_ptr = tl.advance(K_block_ptr, (Bk, 0)), tl.advance(V_block_ptr, (Bk, 0))

    O_i = O_i / l_i[:, None]
    Li = m_i + tl.log(l_i)
    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(L_block_ptr, Li.to(L_block_ptr.type.element_ty), boundary_check=(0,))


@triton.jit
def flash_bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    N_QUERIES, N_KEYS, scale,
    D_MODEL: tl.constexpr, Bq: tl.constexpr, Bk: tl.constexpr,
    is_causal: tl.constexpr,
):
    key_tile_index, batch_index = tl.program_id(0), tl.program_id(1)
    k_start = key_tile_index * Bk

    K_block_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb, shape=(N_KEYS, D_MODEL), strides=(stride_kk, stride_kd), offsets=(k_start, 0), block_shape=(Bk, D_MODEL), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb, shape=(N_KEYS, D_MODEL), strides=(stride_vk, stride_vd), offsets=(k_start, 0), block_shape=(Bk, D_MODEL), order=(1, 0))
    dK_block_ptr = tl.make_block_ptr(dK_ptr + batch_index * stride_dkb, shape=(N_KEYS, D_MODEL), strides=(stride_dkk, stride_dkd), offsets=(k_start, 0), block_shape=(Bk, D_MODEL), order=(1, 0))
    dV_block_ptr = tl.make_block_ptr(dV_ptr + batch_index * stride_dvb, shape=(N_KEYS, D_MODEL), strides=(stride_dvk, stride_dvd), offsets=(k_start, 0), block_shape=(Bk, D_MODEL), order=(1, 0))

    Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dKj, dVj = tl.zeros((Bk, D_MODEL), dtype=tl.float32), tl.zeros((Bk, D_MODEL), dtype=tl.float32)

    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb, shape=(N_QUERIES, D_MODEL), strides=(stride_qq, stride_qd), offsets=(0, 0), block_shape=(Bq, D_MODEL), order=(1, 0))
    dO_block_ptr = tl.make_block_ptr(dO_ptr + batch_index * stride_dob, shape=(N_QUERIES, D_MODEL), strides=(stride_doq, stride_dod), offsets=(0, 0), block_shape=(Bq, D_MODEL), order=(1, 0))
    L_block_ptr = tl.make_block_ptr(L_ptr + batch_index * stride_lb, shape=(N_QUERIES,), strides=(stride_lq,), offsets=(0,), block_shape=(Bq,), order=(0,))
    D_block_ptr = tl.make_block_ptr(D_ptr + batch_index * stride_db, shape=(N_QUERIES,), strides=(stride_dq,), offsets=(0,), block_shape=(Bq,), order=(0,))

    start_i = k_start // Bq if is_causal else 0
    num_q_tiles = tl.cdiv(N_QUERIES, Bq)
    Q_block_ptr = tl.advance(Q_block_ptr, (start_i * Bq, 0))
    dO_block_ptr = tl.advance(dO_block_ptr, (start_i * Bq, 0))
    L_block_ptr = tl.advance(L_block_ptr, (start_i * Bq,))
    D_block_ptr = tl.advance(D_block_ptr, (start_i * Bq,))

    for i in range(start_i, num_q_tiles):
        q_start = i * Bq
        Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        if is_causal:
            causal_mask = (q_start + tl.arange(0, Bq))[:, None] >= (k_start + tl.arange(0, Bk))[None, :]
            Sij = tl.where(causal_mask, Sij, float("-inf"))
        Pij = tl.exp(Sij - Li[:, None])
        dVj += tl.dot(tl.trans(Pij), dOi)
        dPij = tl.dot(dOi, tl.trans(Vj))
        dSij = Pij * (dPij - Di[:, None])
        if is_causal:
            dSij = tl.where(causal_mask, dSij, 0.0)
        dKj += tl.dot(tl.trans(dSij), Qi) * scale
        Q_block_ptr = tl.advance(Q_block_ptr, (Bq, 0))
        dO_block_ptr = tl.advance(dO_block_ptr, (Bq, 0))
        L_block_ptr = tl.advance(L_block_ptr, (Bq,))
        D_block_ptr = tl.advance(D_block_ptr, (Bq,))

    tl.store(dK_block_ptr, dKj.to(dK_block_ptr.type.element_ty), boundary_check=(0, 1))
    tl.store(dV_block_ptr, dVj.to(dV_block_ptr.type.element_ty), boundary_check=(0, 1))


@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr, L_ptr, D_ptr, dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_dob, stride_doq, stride_dod,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    N_QUERIES, N_KEYS, scale,
    D_MODEL: tl.constexpr, Bq: tl.constexpr, Bk: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index, batch_index = tl.program_id(0), tl.program_id(1)
    q_start = query_tile_index * Bq

    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb, shape=(N_QUERIES, D_MODEL), strides=(stride_qq, stride_qd), offsets=(q_start, 0), block_shape=(Bq, D_MODEL), order=(1, 0))
    dO_block_ptr = tl.make_block_ptr(dO_ptr + batch_index * stride_dob, shape=(N_QUERIES, D_MODEL), strides=(stride_doq, stride_dod), offsets=(q_start, 0), block_shape=(Bq, D_MODEL), order=(1, 0))
    L_block_ptr = tl.make_block_ptr(L_ptr + batch_index * stride_lb, shape=(N_QUERIES,), strides=(stride_lq,), offsets=(q_start,), block_shape=(Bq,), order=(0,))
    D_block_ptr = tl.make_block_ptr(D_ptr + batch_index * stride_db, shape=(N_QUERIES,), strides=(stride_dq,), offsets=(q_start,), block_shape=(Bq,), order=(0,))
    dQ_block_ptr = tl.make_block_ptr(dQ_ptr + batch_index * stride_dqb, shape=(N_QUERIES, D_MODEL), strides=(stride_dqq, stride_dqd), offsets=(q_start, 0), block_shape=(Bq, D_MODEL), order=(1, 0))

    Qi = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    dOi = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    Li = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    Di = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    dQi = tl.zeros((Bq, D_MODEL), dtype=tl.float32)

    K_block_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb, shape=(N_KEYS, D_MODEL), strides=(stride_kk, stride_kd), offsets=(0, 0), block_shape=(Bk, D_MODEL), order=(1, 0))
    V_block_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb, shape=(N_KEYS, D_MODEL), strides=(stride_vk, stride_vd), offsets=(0, 0), block_shape=(Bk, D_MODEL), order=(1, 0))

    num_k_tiles = tl.cdiv(q_start + Bq, Bk) if is_causal else tl.cdiv(N_KEYS, Bk)
    for j in range(num_k_tiles):
        k_start = j * Bk
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        Sij = tl.dot(Qi, tl.trans(Kj)) * scale
        if is_causal:
            causal_mask = (q_start + tl.arange(0, Bq))[:, None] >= (k_start + tl.arange(0, Bk))[None, :]
            Sij = tl.where(causal_mask, Sij, float("-inf"))
        Pij = tl.exp(Sij - Li[:, None])
        dPij = tl.dot(dOi, tl.trans(Vj))
        dSij = Pij * (dPij - Di[:, None])
        if is_causal:
            dSij = tl.where(causal_mask, dSij, 0.0)
        dQi += tl.dot(dSij, Kj) * scale
        K_block_ptr, V_block_ptr = tl.advance(K_block_ptr, (Bk, 0)), tl.advance(V_block_ptr, (Bk, 0))

    tl.store(dQ_block_ptr, dQi.to(dQ_block_ptr.type.element_ty), boundary_check=(0, 1))


@functools.lru_cache(maxsize=8)
def _get_shared_mem_limit(device_index: int = 0) -> int:
    props = torch.cuda.get_device_properties(device_index)
    limit = getattr(props, "shared_memory_per_block_optin", None)
    if limit is None or limit == 0:
        limit = getattr(props, "shared_memory_per_block", None)
    if limit is None or limit == 0:
        limit = 48 * 1024
    return int(limit)

def pick_tile_sizes(
    d_model: int, dtype: torch.dtype, device_index: int = 0,
    num_extra_buffers: int = 1, safety_margin: float = 0.75,
) -> tuple[int, int]:
    bytes_per_elem = 2 if dtype in (torch.float16, torch.bfloat16) else 4
    budget = _get_shared_mem_limit(device_index) * safety_margin
    candidates = [128, 64, 32, 16]
    def estimate_bytes(Bq: int, Bk: int) -> int:
        return int((Bq * d_model + 2 * Bk * d_model + Bq * Bk) * bytes_per_elem * num_extra_buffers)
    best = None
    for Bq in candidates:
        for Bk in candidates:
            memory = estimate_bytes(Bq, Bk)
            if memory <= budget:
                score = Bq * Bk
                if best is None or score > best[0]:
                    best = (score, Bq, Bk)
    return (best[1], best[2]) if best is not None else (16, 16)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, is_causal: bool = False):
        if not Q.is_cuda or not K.is_cuda or not V.is_cuda:
            raise RuntimeError("FlashAttentionTriton requires CUDA for all tensors.")
        if Q.dtype not in (torch.float16, torch.bfloat16):
            raise TypeError("FlashAttentionTriton supports float16 and bfloat16.")
        if K.dtype != Q.dtype or V.dtype != Q.dtype:
            raise TypeError("Q/K/V must have the same dtype.")
        if Q.shape[-1] != K.shape[-1] or K.shape[-1] != V.shape[-1]:
            raise ValueError("Head dimensions must match.")
        *batch_dims, Nq, d = Q.shape
        Nk = K.shape[-2]
        scale = 1.0 / math.sqrt(d)
        Q_ = Q.reshape(-1, Nq, d).contiguous()
        K_ = K.reshape(-1, Nk, d).contiguous()
        V_ = V.reshape(-1, Nk, d).contiguous()
        B = Q_.shape[0]
        O_ = torch.empty_like(Q_)
        L_ = torch.empty(B, Nq, device=Q.device, dtype=Q.dtype)
        device_index = Q.device.index if Q.device.index is not None else 0
        Bq, Bk = pick_tile_sizes(d_model=d, dtype=Q.dtype, device_index=device_index, num_extra_buffers=3)
        Tq = triton.cdiv(Nq, Bq)
        flash_fwd_kernel[(Tq, B)](
            Q_, K_, V_, O_, L_,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            O_.stride(0), O_.stride(1), O_.stride(2),
            L_.stride(0), L_.stride(1),
            Nq, Nk, scale,
            D=d, Bq=Bq, Bk=Bk,
            is_causal=is_causal,
        )
        O = O_.reshape(*batch_dims, Nq, d)
        L = L_.reshape(*batch_dims, Nq)
        ctx.save_for_backward(L, Q, K, V, O)
        ctx.is_causal = is_causal
        return O


    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        *batch_dims, Nq, d = Q.shape
        Nk = K.shape[-2]
        scale = 1.0 / math.sqrt(d)
        Q_ = Q.reshape(-1, Nq, d).contiguous()
        K_ = K.reshape(-1, Nk, d).contiguous()
        V_ = V.reshape(-1, Nk, d).contiguous()
        O_ = O.reshape(-1, Nq, d).contiguous()
        L_ = L.reshape(-1, Nq).contiguous()
        dO_ = grad_out.reshape(-1, Nq, d).contiguous()
        B = Q_.shape[0]
        D_ = (dO_ * O_).sum(dim=-1).contiguous()
        dQ_, dK_, dV_ = torch.zeros_like(Q_), torch.zeros_like(K_), torch.zeros_like(V_)
        device_index = Q.device.index if Q.device.index is not None else 0
        Bq, Bk = pick_tile_sizes(d_model=d, dtype=Q.dtype, device_index=device_index, num_extra_buffers=3)
        Tq, Tk = triton.cdiv(Nq, Bq), triton.cdiv(Nk, Bk)
        flash_bwd_dkdv_kernel[(Tk, B)](
            Q_, K_, V_, dO_, L_, D_, dK_, dV_,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            dO_.stride(0), dO_.stride(1), dO_.stride(2),
            L_.stride(0), L_.stride(1),
            D_.stride(0), D_.stride(1),
            dK_.stride(0), dK_.stride(1), dK_.stride(2),
            dV_.stride(0), dV_.stride(1), dV_.stride(2),
            Nq, Nk, scale,
            D_MODEL=d, Bq=Bq, Bk=Bk,
            is_causal=is_causal,
        )
        flash_bwd_dq_kernel[(Tq, B)](
            Q_, K_, V_, dO_, L_, D_, dQ_,
            Q_.stride(0), Q_.stride(1), Q_.stride(2),
            K_.stride(0), K_.stride(1), K_.stride(2),
            V_.stride(0), V_.stride(1), V_.stride(2),
            dO_.stride(0), dO_.stride(1), dO_.stride(2),
            L_.stride(0), L_.stride(1),
            D_.stride(0), D_.stride(1),
            dQ_.stride(0), dQ_.stride(1), dQ_.stride(2),
            Nq, Nk, scale,
            D_MODEL=d, Bq=Bq, Bk=Bk,
            is_causal=is_causal,
        )
        return (dQ_.reshape(*batch_dims, Nq, d),
                dK_.reshape(*batch_dims, Nk, d),
                dV_.reshape(*batch_dims, Nk, d),
                None)