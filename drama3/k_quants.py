"""Q2_K / Q3_K experts: GGML's 84 / 110-byte blocks, BF16 compute.

CUDA decode reads packed weights directly; prefill dequantizes only the expert
being used. CPU decoding is a reference fallback. No GGUF runtime is required.
The block interpretation follows llama.cpp (MIT), ggml-common.h/ggml-quants.c.
"""
import torch
from torch import nn

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

BLOCK_WEIGHTS = 256
BLOCK_BYTES = {2: 84, 3: 110}


def dequantize_cpu(packed, columns, dtype=torch.bfloat16, bits=3):
    if packed.dtype != torch.uint8 or columns % BLOCK_WEIGHTS:
        raise ValueError("K-quants need uint8 blocks and columns divisible by 256")
    blocks = packed.reshape(-1, columns // 256, BLOCK_BYTES[bits])
    j = torch.arange(256, device=packed.device)
    group = j // 16
    if bits == 2:
        codes = (blocks[..., 16 + (j // 128) * 32 + j % 32].int() >> ((j % 128 // 32) * 2)) & 3
        scales = blocks[..., group].int()
        d = blocks[..., 80:82].contiguous().view(torch.float16).squeeze(-1).float()
        dmin = blocks[..., 82:84].contiguous().view(torch.float16).squeeze(-1).float()
        return (d[..., None] * (scales & 15) * codes - dmin[..., None] * (scales >> 4)).reshape(-1, columns).to(dtype)
    low = (blocks[..., 32 + (j // 128) * 32 + j % 32].int() >> ((j % 128 // 32) * 2)) & 3
    high = (blocks[..., j % 32].int() >> (j // 32)) & 1
    scale_low = (blocks[..., 96 + group % 8].int() >> ((group // 8) * 4)) & 15
    scale_high = (blocks[..., 104 + group % 4].int() >> ((group // 4) * 2)) & 3
    scale = (scale_low | (scale_high << 4)) - 32
    d = blocks[..., 108:110].contiguous().view(torch.float16).squeeze(-1).float()
    return (d[..., None] * scale * (low - 4 * (1 - high))).reshape(-1, columns).to(dtype)


if triton is not None:
    @triton.jit
    def _weights(W, row, col, K: tl.constexpr, BITS: tl.constexpr):
        local = col % 256
        group = local // 16
        if BITS == 2:
            base = row[:, None] * (K // 256 * 84) + (col[None, :] // 256) * 84
            codes = (tl.load(W + base + 16 + (local[None, :] // 128) * 32 + local[None, :] % 32).to(tl.int32)
                     >> ((local[None, :] % 128 // 32) * 2)) & 3
            scales = tl.load(W + base + group[None, :]).to(tl.int32)
            d = tl.load((W + base + 80).to(tl.pointer_type(tl.float16))).to(tl.float32)
            dmin = tl.load((W + base + 82).to(tl.pointer_type(tl.float16))).to(tl.float32)
            return d * (scales & 15).to(tl.float32) * codes.to(tl.float32) - dmin * (scales >> 4).to(tl.float32)
        base = row[:, None] * (K // 256 * 110) + (col[None, :] // 256) * 110
        low = (tl.load(W + base + 32 + (local[None, :] // 128) * 32 + local[None, :] % 32).to(tl.int32)
               >> ((local[None, :] % 128 // 32) * 2)) & 3
        high = (tl.load(W + base + local[None, :] % 32).to(tl.int32) >> (local[None, :] // 32)) & 1
        sl = (tl.load(W + base + 96 + group[None, :] % 8).to(tl.int32) >> ((group[None, :] // 8) * 4)) & 15
        sh = (tl.load(W + base + 104 + group[None, :] % 4).to(tl.int32) >> ((group[None, :] // 4) * 2)) & 3
        d = tl.load((W + base + 108).to(tl.pointer_type(tl.float16))).to(tl.float32)
        return d * ((sl | (sh << 4)) - 32).to(tl.float32) * (low - 4 * (1 - high)).to(tl.float32)

    @triton.jit
    def _gemv(X, W, Y, K: tl.constexpr, ROWS_PER_INPUT: tl.constexpr, BN: tl.constexpr, BITS: tl.constexpr):
        row = tl.program_id(0) * BN + tl.arange(0, BN)
        col = tl.arange(0, K)
        x = tl.load(X + (row[:, None] // ROWS_PER_INPUT) * K + col[None, :])
        w = _weights(W, row, col, K, BITS).to(x.dtype).to(tl.float32)
        result = tl.sum(w * x.to(tl.float32), axis=1)
        tl.store(Y + row, result)

    @triton.jit
    def _dequantize(W, Y, K: tl.constexpr, BN: tl.constexpr, BITS: tl.constexpr):
        row = tl.program_id(0) * BN + tl.arange(0, BN)
        col = tl.arange(0, K)
        tl.store(Y + row[:, None] * K + col[None, :], _weights(W, row, col, K, BITS))


def dequantize(packed, columns, dtype=torch.bfloat16, bits=3):
    if packed.device.type != "cuda":
        return dequantize_cpu(packed, columns, dtype, bits)
    if triton is None:
        raise RuntimeError("Q2_K / Q3_K CUDA execution requires Triton")
    rows = packed.shape[0]
    output = torch.empty((rows, columns), device=packed.device, dtype=dtype)
    if rows % 4:
        raise ValueError("K-quant expert rows must be divisible by four")
    _dequantize[(rows // 4,)](packed, output, columns, 4, bits, num_warps=4, enable_fp_fusion=bits != 2)
    return output


def linear(x, packed, columns, rows_per_input=None, bits=3):
    """A linear projection, optionally one input row per expert's output rows."""
    if x.ndim != 2 or x.shape[-1] != columns or packed.ndim != 2:
        raise ValueError("Invalid K-quant projection dimensions")
    rows = packed.shape[0]
    if packed.shape[1] != columns // 256 * BLOCK_BYTES[bits] or not packed.is_contiguous():
        raise ValueError("Invalid K-quant packed row layout")
    if x.device != packed.device:
        raise ValueError("K-quant activations and weights must be on the same device")
    if rows_per_input is not None and rows != x.shape[0] * rows_per_input:
        raise ValueError("Grouped K-quant projection needs one activation per expert")
    if x.device.type == "cuda" and (x.shape[0] == 1 or rows_per_input is not None):
        if triton is None:
            raise RuntimeError("Q2_K / Q3_K CUDA execution requires Triton")
        output = torch.empty(rows, device=x.device, dtype=x.dtype)
        if rows % 4:
            raise ValueError("K-quant expert rows must be divisible by four")
        _gemv[(rows // 4,)](x.contiguous(), packed, output, columns, rows_per_input or rows, 4, bits,
                           num_warps=4, enable_fp_fusion=bits != 2)
        return output.view(x.shape[0], rows_per_input) if rows_per_input else output[None]
    weight = dequantize(packed, columns, x.dtype, bits)
    if rows_per_input is None:
        return nn.functional.linear(x, weight)
    return torch.bmm(weight.view(x.shape[0], rows_per_input, columns), x[:, :, None]).squeeze(-1)


class KQuantExperts(nn.Module):
    def __init__(self, num_experts, hidden, intermediate, device=None, bits=3):
        super().__init__()
        self.num_experts = num_experts
        self.hidden = hidden
        self.intermediate = intermediate
        self.bits = bits
        for name, rows, cols in (("gate_up", 2 * intermediate, hidden), ("down", hidden, intermediate)):
            self.register_buffer(name + f"_q{bits}k", torch.empty((num_experts, rows, cols // 256 * BLOCK_BYTES[bits]),
                                                           dtype=torch.uint8, device=device))

    def forward(self, *args):
        raise RuntimeError("Q2_K / Q3_K Drama 3 requires the Load Drama 3 node's experts='in RAM' mode")
