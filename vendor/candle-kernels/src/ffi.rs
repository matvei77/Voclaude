//! Stubs for the mixture-of-experts kernels that upstream candle-kernels
//! links from `libmoe.a`. Those CUDA sources need bf16 tensor-core fragments
//! (compute capability 8.0+) and cannot be built for the 7.5 target Voclaude
//! ships; the app never runs an MoE model, so the entry points only need to
//! exist for candle-nn's `moe` module to link.
use core::ffi::c_void;

const MSG: &str = "MoE kernels are not built into this binary";

/// # Safety
/// Never dereferences its arguments; always panics.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn moe_gemm_wmma(
    _input: *const c_void,
    _weights: *const c_void,
    _sorted_token_ids: *const i32,
    _expert_ids: *const i32,
    _topk_weights: *const f32,
    _output: *mut c_void,
    _expert_counts: *mut i32,
    _expert_offsets: *mut i32,
    _num_experts: i32,
    _topk: i32,
    _size_m: i32,
    _size_n: i32,
    _size_k: i32,
    _dtype: i32,
    _is_prefill: bool,
    _stream: i64,
) {
    panic!("{}", MSG)
}

/// # Safety
/// Never dereferences its arguments; always panics.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn moe_gemm_gguf(
    _input: *const f32,
    _weights: *const c_void,
    _sorted_token_ids: *const i32,
    _expert_ids: *const i32,
    _topk_weights: *const f32,
    _output: *mut c_void,
    _num_experts: i32,
    _topk: i32,
    _size_m: i32,
    _size_n: i32,
    _size_k: i32,
    _gguf_dtype: i32,
    _stream: i64,
) {
    panic!("{}", MSG)
}

/// # Safety
/// Never dereferences its arguments; always panics.
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn moe_gemm_gguf_prefill(
    _input: *const c_void,
    _weights: *const u8,
    _sorted_token_ids: *const i32,
    _expert_ids: *const i32,
    _topk_weights: *const f32,
    _output: *mut c_void,
    _num_experts: i32,
    _topk: i32,
    _size_m: i32,
    _size_n: i32,
    _size_k: i32,
    _input_dtype: i32,
    _gguf_dtype: i32,
    _stream: i64,
) {
    panic!("{}", MSG)
}
