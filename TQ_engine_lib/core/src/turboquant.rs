use numpy::{PyArray, PyArrayMethods, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
use numpy::ndarray::Axis;

// =============================================================================
// Quantization helpers (index-time)
// - SQ bits supported: 1 or 3 (TurboQuant 2b/4b)
// - Produces:
//   - packed SQ codes (uint8)
//   - packed QJL signs (uint8, packbits little across dim)
//   - res_norms (float32): L2 norm of residual (x_rot - x_hat_1)
// =============================================================================

#[pyfunction]
pub fn tq_quantize_rotated(
    py: Python<'_>,
    x_rot: Bound<'_, PyArray2<f32>>,        // (N, D) rotated residual-space vectors
    sq_centroids: Bound<'_, PyArray1<f32>>, // (K,) scalar codebook (K=2 or 8)
    sq_bits: i32,                           // 1 or 3
) -> PyResult<(Py<PyArray2<u8>>, Py<PyArray2<u8>>, Py<PyArray1<f32>>)> {
    let x_ro = x_rot.readonly();
    let x = x_ro.as_array();
    let n = x.shape()[0];
    let d = x.shape()[1];

    let cent_ro = sq_centroids.readonly();
    let cent = cent_ro.as_slice()?;

    let bits = sq_bits as usize;
    if bits != 1 && bits != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("tq_quantize_rotated only supports sq_bits=1 or 3 (got {})", bits),
        ));
    }

    let k = 1usize << bits;
    if cent.len() != k {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("sq_centroids length must be {} for sq_bits={}, got {}", k, bits, cent.len()),
        ));
    }

    // boundaries: midpoints between centroids (sorted). Works for both 2 and 8 levels.
    let mut boundaries = vec![0.0f32; k + 1];
    boundaries[0] = f32::NEG_INFINITY;
    boundaries[k] = f32::INFINITY;
    for i in 0..(k - 1) {
        boundaries[i + 1] = 0.5 * (cent[i] + cent[i + 1]);
    }

    // output shapes
    let vals_per_byte = if bits == 1 { 8usize } else { 2usize }; // 1b: 8 vals/byte, 3b: 2 vals/byte
    let packed_sq_d = (d + vals_per_byte - 1) / vals_per_byte;
    let packed_qjl_d = (d + 8 - 1) / 8;

    let mut sq_codes = vec![0u8; n * packed_sq_d];
    let mut qjl_signs = vec![0u8; n * packed_qjl_d];
    let mut res_norms = vec![0.0f32; n];

    // helper: find quant bin using boundaries (linear scan; k small)
    #[inline(always)]
    fn quant_bin(v: f32, boundaries: &[f32]) -> usize {
        // boundaries length = K+1; find i such that boundaries[i] <= v < boundaries[i+1]
        // K is 2 or 8 -> cheap.
        let mut lo = 0usize;
        let mut hi = boundaries.len() - 1;
        // binary search on boundaries
        while lo + 1 < hi {
            let mid = (lo + hi) >> 1;
            if v >= boundaries[mid] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    py.allow_threads(|| {
        sq_codes
            .par_chunks_mut(packed_sq_d)
            .zip(qjl_signs.par_chunks_mut(packed_qjl_d))
            .zip(res_norms.par_iter_mut())
            .enumerate()
            .for_each(|(row, ((sq_row, qjl_row), rn_out))| {
                // zero rows (par_chunks_mut gives fresh chunks but we ensure)
                for b in sq_row.iter_mut() {
                    *b = 0;
                }
                for b in qjl_row.iter_mut() {
                    *b = 0;
                }

                let mut sum_sq = 0.0f32;
                for j in 0..d {
                    let v = x[[row, j]];
                    let qi = quant_bin(v, &boundaries); // 0..K-1
                    let xhat = cent[qi];
                    let r = v - xhat;
                    sum_sq += r * r;

                    // SQ pack
                    if bits == 1 {
                        // 1 bit per dim: qi is 0/1
                        let byte = j >> 3;
                        let bit = j & 7;
                        sq_row[byte] |= ((qi as u8) & 1) << bit;
                    } else {
                        // 3 bits per dim, pack 2 dims per byte (little packing: dim0 in low bits)
                        let byte = j >> 1;
                        let shift = (j & 1) * 3;
                        sq_row[byte] |= (qi as u8) << shift;
                    }

                    // QJL sign of residual in rotated space: residual > 0
                    let s = (r > 0.0) as u8;
                    let qbyte = j >> 3;
                    let qbit = j & 7;
                    qjl_row[qbyte] |= (s & 1) << qbit;
                }
                *rn_out = sum_sq.sqrt();
            });
    });

    let sq_arr = PyArray1::from_vec_bound(py, sq_codes).reshape([n, packed_sq_d])?;
    let qjl_arr = PyArray1::from_vec_bound(py, qjl_signs).reshape([n, packed_qjl_d])?;
    let rn_arr = PyArray1::from_vec_bound(py, res_norms);
    Ok((sq_arr.unbind(), qjl_arr.unbind(), rn_arr.unbind()))
}


// =============================================================================
// TQ Unified Scan — SQ(b-1) + QJL(1)
// Tối ưu cho Normalized Vectors (Dot Product = Cosine Similarity)
// =============================================================================

#[pyfunction]
pub fn tq_scan(
    py: Python<'_>,
    query: Bound<'_, PyArray1<f32>>,
    sq_codes: Bound<'_, PyArray2<u8>>,
    centroids: Bound<'_, PyArray1<f32>>,
    norms: Bound<'_, PyArray1<f32>>,
    qjl_signs: Bound<'_, PyArray2<u8>>,
    res_norms: Bound<'_, PyArray1<f32>>,
    qjl_query: Bound<'_, PyArray1<f32>>,
    qjl_scale: f32,
    dim: i32,
    mse_bits: i32,
) -> PyResult<PyObject> {
    let d = dim as usize;
    let query_readonly = query.readonly();
    let q_slice = query_readonly.as_slice()?;
    
    let centroids_readonly = centroids.readonly();
    let centroids_slice = centroids_readonly.as_slice()?;
    
    let norms_readonly = norms.readonly();
    let norms_slice = norms_readonly.as_slice()?;
    
    let res_norms_readonly = res_norms.readonly();
    let res_norms_slice = res_norms_readonly.as_slice()?;
    
    let qjl_query_readonly = qjl_query.readonly();
    let qjl_q_slice = qjl_query_readonly.as_slice()?;
    
    let sq_codes_readonly = sq_codes.readonly();
    let sq_codes_view = sq_codes_readonly.as_array();
    let n = sq_codes_view.shape()[0];
    let packed_d = sq_codes_view.shape()[1];
    
    let qjl_signs_readonly = qjl_signs.readonly();
    let qjl_signs_view = qjl_signs_readonly.as_array();
    let qjl_dim = qjl_q_slice.len();
    
    // Bit packing config
    let bits = mse_bits as u32;
    let bit_mask = (1u32 << bits) - 1;

    // VALIDATION: TurboQuant Native only supports 1-bit and 3-bit MSE
    if bits != 1 && bits != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("TurboQuant Native only supports 1-bit and 3-bit MSE (received {} bits)", bits)
        ));
    }

    let mut output = vec![0.0f32; n];
    let sq_slice_flat = sq_codes_view.as_slice().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("SQ codes must be contiguous"))?;
    let qjl_signs_flat = qjl_signs_view.as_slice().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("QJL signs must be contiguous"))?;
    
    py.allow_threads(|| {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(8).build().unwrap();
        let chunk_size = 8192;
        
        pool.install(|| {
            output.par_chunks_mut(chunk_size).enumerate().for_each(|(chunk_idx, chunk)| {
                let start_idx = chunk_idx * chunk_size;
                use std::arch::x86_64::*;

                unsafe {
                    let v_bit_indices = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
                    let v_one = _mm256_set1_ps(1.0);
                    let v_neg_one = _mm256_set1_ps(-1.0);
                    
                    for (sub_idx, score) in chunk.iter_mut().enumerate() {
                        let i = start_idx + sub_idx;
                        if i >= n { break; }
                        
                            let packed_sq_d = if mse_bits == 1 { d / 8 } else { d / 2 };
                            let row_sq_start = i * packed_sq_d;
                        let row_qjl_start = i * (qjl_dim / 8);
                        
                        // --- STAGE 1: SQ ---
                        let mut current_dot: f32 = 0.0;
                        if bits == 1 {
                            let mut v_acc = _mm256_setzero_ps();
                            let v_pos = _mm256_set1_ps(centroids_slice[1]);
                            let v_neg = _mm256_set1_ps(centroids_slice[0]);
                            for k in (0..d).step_by(8) {
                                let b = sq_slice_flat[row_sq_start + (k / 8)];
                                let v_b = _mm256_set1_epi32(b as i32);
                                let v_mask = _mm256_cmpeq_epi32(_mm256_and_si256(v_b, v_bit_indices), v_bit_indices);
                                let v_q = _mm256_loadu_ps(q_slice.as_ptr().add(k));
                                let v_k = _mm256_blendv_ps(v_neg, v_pos, _mm256_castsi256_ps(v_mask));
                                v_acc = _mm256_fmadd_ps(v_q, v_k, v_acc);
                            }
                            let mut tmp = [0.0f32; 8];
                            _mm256_storeu_ps(tmp.as_mut_ptr(), v_acc);
                            current_dot = tmp.iter().sum::<f32>();
                        } else {
                            let mut v_acc = _mm256_setzero_ps();
                            for k in (0..d).step_by(8) {
                                let b0 = sq_slice_flat[row_sq_start + (k / 2)];
                                let b1 = sq_slice_flat[row_sq_start + (k / 2 + 1)];
                                let b2 = sq_slice_flat[row_sq_start + (k / 2 + 2)];
                                let b3 = sq_slice_flat[row_sq_start + (k / 2 + 3)];
                                let indices = [
                                    (b0 & 0x7) as i32, ((b0 >> 3) & 0x7) as i32,
                                    (b1 & 0x7) as i32, ((b1 >> 3) & 0x7) as i32,
                                    (b2 & 0x7) as i32, ((b2 >> 3) & 0x7) as i32,
                                    (b3 & 0x7) as i32, ((b3 >> 3) & 0x7) as i32,
                                ];
                                let v_indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
                                let v_k = _mm256_i32gather_ps(centroids_slice.as_ptr(), v_indices, 4);
                                let v_q = _mm256_loadu_ps(q_slice.as_ptr().add(k));
                                v_acc = _mm256_fmadd_ps(v_q, v_k, v_acc);
                            }
                            let mut tmp = [0.0f32; 8];
                            _mm256_storeu_ps(tmp.as_mut_ptr(), v_acc);
                            current_dot = tmp.iter().sum::<f32>();
                        }

                        // --- STAGE 2: QJL ---
                        let mut v_sum = _mm256_setzero_ps();
                        for k in (0..qjl_dim).step_by(8) {
                            let b = qjl_signs_flat[row_qjl_start + (k / 8)];
                            let v_b = _mm256_set1_epi32(b as i32);
                            let v_mask = _mm256_cmpeq_epi32(_mm256_and_si256(v_b, v_bit_indices), v_bit_indices);
                            let v_q = _mm256_loadu_ps(qjl_q_slice.as_ptr().add(k));
                            let v_sign = _mm256_blendv_ps(v_neg_one, v_one, _mm256_castsi256_ps(v_mask));
                            v_sum = _mm256_fmadd_ps(v_q, v_sign, v_sum);
                        }
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), v_sum);
                        let qjl_corr = tmp.iter().sum::<f32>() * qjl_scale * res_norms_slice[i];
                        
                        *score = (current_dot + qjl_corr) * norms_slice[i];
                    }
                }
            });
        });
    });

    let array = PyArray::from_vec_bound(py, output);
    Ok(array.into())
}


#[pyfunction]
pub fn tq_batch_scan(
    py: Python<'_>,
    queries: Bound<'_, PyArray2<f32>>,
    sq_codes: Bound<'_, PyArray2<u8>>,
    centroids: Bound<'_, PyArray1<f32>>,
    norms: Bound<'_, PyArray1<f32>>,
    qjl_signs: Bound<'_, PyArray2<u8>>,
    res_norms: Bound<'_, PyArray1<f32>>,
    qjl_queries: Bound<'_, PyArray2<f32>>,
    qjl_scale: f32,
    dim: i32,
    mse_bits: i32,
) -> PyResult<PyObject> {
    let d = dim as usize;
    let queries_readonly = queries.readonly();
    let queries_view = queries_readonly.as_array();
    let num_queries = queries_view.shape()[0];
    
    let qjl_queries_readonly = qjl_queries.readonly();
    let qjl_queries_view = qjl_queries_readonly.as_array();
    
    let centroids_readonly = centroids.readonly();
    let centroids_slice = centroids_readonly.as_slice()?;
    
    let norms_readonly = norms.readonly();
    let norms_slice = norms_readonly.as_slice()?;
    
    let res_norms_readonly = res_norms.readonly();
    let res_norms_slice = res_norms_readonly.as_slice()?;
    
    let sq_codes_readonly = sq_codes.readonly();
    let sq_codes_view = sq_codes_readonly.as_array();
    let n = sq_codes_view.shape()[0];
    let packed_d = sq_codes_view.shape()[1];
    
    let qjl_signs_readonly = qjl_signs.readonly();
    let qjl_signs_view = qjl_signs_readonly.as_array();
    let qjl_dim = qjl_queries_view.shape()[1];
    
    let bits = mse_bits as u32;
    let mut output = vec![0.0f32; num_queries * n];
    
    let sq_slice_flat = sq_codes_view.as_slice().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("SQ codes must be contiguous"))?;
    let qjl_signs_flat = qjl_signs_view.as_slice().ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("QJL signs must be contiguous"))?;

    py.allow_threads(|| {
        output.par_chunks_mut(n).enumerate().for_each(|(q_idx, scores)| {
            let q_row = queries_view.index_axis(Axis(0), q_idx);
            let q_slice = q_row.as_slice().unwrap();
            
            let qjl_q_row = qjl_queries_view.index_axis(Axis(0), q_idx);
            let qjl_q_slice = qjl_q_row.as_slice().unwrap();

            
            use std::arch::x86_64::*;
            unsafe {
                let v_bit_indices = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
                let v_one = _mm256_set1_ps(1.0);
                let v_neg_one = _mm256_set1_ps(-1.0);
                
                for i in 0..n {
                            let packed_sq_d = if mse_bits == 1 { d / 8 } else { d / 2 };
                            let row_sq_start = i * packed_sq_d;
                    let row_qjl_start = i * (qjl_dim / 8);
                    let mut current_dot: f32 = 0.0;
                    
                    // --- SQ Stage ---
                    if bits == 1 {
                        let mut v_acc = _mm256_setzero_ps();
                        let v_pos = _mm256_set1_ps(centroids_slice[1]);
                        let v_neg = _mm256_set1_ps(centroids_slice[0]);
                        for k in (0..d).step_by(8) {
                            let b = sq_slice_flat[row_sq_start + (k / 8)];
                            let v_b = _mm256_set1_epi32(b as i32);
                            let v_mask = _mm256_cmpeq_epi32(_mm256_and_si256(v_b, v_bit_indices), v_bit_indices);
                            let v_q = _mm256_loadu_ps(q_slice.as_ptr().add(k));
                            let v_k = _mm256_blendv_ps(v_neg, v_pos, _mm256_castsi256_ps(v_mask));
                            v_acc = _mm256_fmadd_ps(v_q, v_k, v_acc);
                        }
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), v_acc);
                        current_dot = tmp.iter().sum::<f32>();
                    } else {
                        let mut v_acc = _mm256_setzero_ps();
                        for k in (0..d).step_by(8) {
                            let b0 = sq_slice_flat[row_sq_start + (k / 2)];
                            let b1 = sq_slice_flat[row_sq_start + (k / 2 + 1)];
                            let b2 = sq_slice_flat[row_sq_start + (k / 2 + 2)];
                            let b3 = sq_slice_flat[row_sq_start + (k / 2 + 3)];
                            let indices = [(b0 & 0x7) as i32, ((b0 >> 3) & 0x7) as i32, (b1 & 0x7) as i32, ((b1 >> 3) & 0x7) as i32, (b2 & 0x7) as i32, ((b2 >> 3) & 0x7) as i32, (b3 & 0x7) as i32, ((b3 >> 3) & 0x7) as i32];
                            let v_indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
                            let v_k = _mm256_i32gather_ps(centroids_slice.as_ptr(), v_indices, 4);
                            let v_q = _mm256_loadu_ps(q_slice.as_ptr().add(k));
                            v_acc = _mm256_fmadd_ps(v_q, v_k, v_acc);
                        }
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), v_acc);
                        current_dot = tmp.iter().sum::<f32>();
                    }

                    // --- QJL Stage ---
                    let mut v_sum = _mm256_setzero_ps();
                    for k in (0..qjl_dim).step_by(8) {
                        let b = qjl_signs_flat[row_qjl_start + (k / 8)];
                        let v_b = _mm256_set1_epi32(b as i32);
                        let v_mask = _mm256_cmpeq_epi32(_mm256_and_si256(v_b, v_bit_indices), v_bit_indices);
                        let v_q = _mm256_loadu_ps(qjl_q_slice.as_ptr().add(k));
                        let v_sign = _mm256_blendv_ps(v_neg_one, v_one, _mm256_castsi256_ps(v_mask));
                        v_sum = _mm256_fmadd_ps(v_q, v_sign, v_sum);
                    }
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), v_sum);
                    let qjl_corr = tmp.iter().sum::<f32>() * qjl_scale * res_norms_slice[i];
                    
                    scores[i] = (current_dot + qjl_corr) * norms_slice[i];
                }
            }
        });
    });

    let array = PyArray1::from_vec_bound(py, output).reshape([num_queries, n])?;
    Ok(array.into())
}

#[pyfunction]
pub fn mse_score_simd(
    py: Python<'_>,
    _query: Bound<'_, PyArray1<f32>>,
    _centroids: Bound<'_, PyArray1<f32>>,
    mse_codes: Bound<'_, PyArray2<u8>>,
    _norms: Bound<'_, PyArray1<f32>>,
    _dim: i32,
    _bits: i32,
) -> PyResult<PyObject> {
    let n = mse_codes.readonly().as_array().shape()[0];
    let output = vec![0.0f32; n];
    Ok(PyArray::from_vec_bound(py, output).into())
}

#[pyfunction]
pub fn tq_master_scan(
    py: Python<'_>,
    q_rot: Bound<'_, PyArray2<f32>>,
    mse_packed: Bound<'_, PyArray3<u8>>,
    _qjl_signs: Bound<'_, PyArray3<u8>>,
    _norms: Bound<'_, PyArray2<f32>>,
    _centroids: Bound<'_, PyArray1<f32>>,
    _mse_bits: i32,
    _qjl_scale: f32,
) -> PyResult<PyObject> {
    let bh = q_rot.readonly().as_array().shape()[0];
    let n = mse_packed.readonly().as_array().shape()[1];
    let output = vec![0.0f32; bh * n];
    let array = PyArray1::from_vec_bound(py, output).reshape([bh, n])?;
    Ok(array.into())
}

// =============================================================================
// IVF Online Batch Scan — SIMD Multi-Query Engine
// Returns shape (num_queries, n_total) score matrix
// =============================================================================

#[inline(always)]
fn float_to_ordered_u32(f: f32) -> u32 {
    let bits = f.to_bits();
    if bits & 0x80000000 != 0 {
        !bits
    } else {
        bits | 0x80000000
    }
}

#[inline(always)]
fn ordered_u32_to_float(u: u32) -> f32 {
    let bits = if u & 0x80000000 == 0 {
        !u
    } else {
        u & 0x7FFFFFFF
    };
    f32::from_bits(bits)
}

#[pyfunction]
pub fn tq_ivf_online_scan(
    py: Python<'_>,
    queries: Bound<'_, PyArray2<f32>>,
    full_sq: Bound<'_, PyArray2<u8>>,
    centroids: Bound<'_, PyArray1<f32>>,
    full_norms: Bound<'_, PyArray1<f32>>,
    full_signs: Bound<'_, PyArray2<u8>>,
    full_res: Bound<'_, PyArray1<f32>>,
    qjl_queries: Bound<'_, PyArray2<f32>>,
    list_offsets: Bound<'_, PyArray1<i32>>,
    coarse_centroids: Bound<'_, PyArray2<f32>>,
    n_probe: usize,
    qjl_scale: f32,
    dim: i32,
    mse_bits: i32,
    top_k: usize,
    
) -> PyResult<(Py<PyArray2<f32>>, Py<PyArray2<i32>>)> {
    use std::sync::Mutex;
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;
    use rayon::prelude::*;

    let d = dim as usize;
    let num_levels = if mse_bits == 1 { 2usize } else { 1usize << mse_bits };

    // --- Read all numpy arrays ---
    let queries_ro = queries.readonly();
    let queries_view = queries_ro.as_array();
    let num_queries = queries_view.shape()[0];

    let qjl_queries_ro = qjl_queries.readonly();
    let qjl_queries_view = qjl_queries_ro.as_array();
    let qjl_dim = qjl_queries_view.shape()[1];

    let coarse_ro = coarse_centroids.readonly();
    let coarse_view = coarse_ro.as_array();
    let num_centroids = coarse_view.shape()[0];

    let offsets_ro = list_offsets.readonly();
    let offsets_sl = offsets_ro.as_slice()?;

    let cent_ro = centroids.readonly();
    let cent_sl = cent_ro.as_slice()?;

    let norms_ro = full_norms.readonly();
    let norms_sl = norms_ro.as_slice()?;

    let res_ro = full_res.readonly();
    let res_sl = res_ro.as_slice()?;

    let sq_ro = full_sq.readonly();
    let sq_view = sq_ro.as_array();
    let sq_flat = sq_view.as_slice()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("SQ codes not contiguous"))?;

    let signs_ro = full_signs.readonly();
    let signs_view = signs_ro.as_array();
    let signs_flat = signs_view.as_slice()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("QJL signs not contiguous"))?;

    // --- Derived constants ---
    let packed_sq_d = if mse_bits == 1 { d / 8 } else { d / 2 };

    // --- PRODUCTION FIX (Option C): Global heaps + Periodic Flush ---
    // Global heaps for each query
    let global_heaps: Vec<Mutex<BinaryHeap<Reverse<(u32, u32)>>>> = (0..num_queries)
        .map(|_| Mutex::new(BinaryHeap::with_capacity(top_k)))
        .collect();

    let mut query_to_clusters = vec![vec![0usize; n_probe]; num_queries];
    // Cache coarse dot-products (q · centroid) for the selected clusters.
    // We compute IPs during Phase 1 routing; reuse them in Phase 3 to avoid an extra O(d)
    // dot-product per (query, cluster) in the hot loop.
    let mut query_to_cluster_ip = vec![vec![0.0f32; n_probe]; num_queries];
    py.allow_threads(|| {
        // Phase 1: Coarse Routing — must match Python IVF assignment (quantizer.index):
        // clusters are chosen by argmax inner product with coarse centroids (DPR / IP),
        // not L2 distance to centroids (those differ when centroid norms vary).
        query_to_clusters
            .par_iter_mut()
            .zip(query_to_cluster_ip.par_iter_mut())
            .enumerate()
            .for_each(|(q_idx, (out, out_ip))| {
            let q_row = queries_view.index_axis(Axis(0), q_idx);
            let q_sl = q_row.as_slice().unwrap();

            // store (-ip, ci, ip) so we can reuse ip for selected clusters
            let mut dists: Vec<(f32, usize, f32)> = (0..num_centroids).map(|ci| {
                let c_row = coarse_view.index_axis(Axis(0), ci);
                let c_sl = c_row.as_slice().unwrap();
                let ip = unsafe {
                    use std::arch::x86_64::*;
                    let mut v_sum = _mm256_setzero_ps();
                    for k in (0..d).step_by(8) {
                        let vq = _mm256_loadu_ps(q_sl.as_ptr().add(k));
                        let vc = _mm256_loadu_ps(c_sl.as_ptr().add(k));
                        v_sum = _mm256_fmadd_ps(vq, vc, v_sum);
                    }
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), v_sum);
                    tmp.iter().sum::<f32>()
                };
                // Keep smallest keys = largest IP (same partition logic as L2^2 before).
                (-ip, ci, ip)
            }).collect();

            let actual_probe = n_probe.min(num_centroids);
            if actual_probe < num_centroids {
                dists.select_nth_unstable_by(actual_probe, |a, b| a.0.partial_cmp(&b.0).unwrap());
            }
            for i in 0..actual_probe {
                out[i] = dists[i].1;
                out_ip[i] = dists[i].2;
            }
        });

        // Phase 2: Grouping (Sequential but fast)
        let mut cluster_to_queries: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_centroids];
        for q_idx in 0..num_queries {
            for j in 0..query_to_clusters[q_idx].len() {
                let c_idx = query_to_clusters[q_idx][j];
                let ip = query_to_cluster_ip[q_idx][j];
                cluster_to_queries[c_idx].push((q_idx, ip));
            }
        }

        // Phase 3: Parallel Scan with Hybrid Top-K
        cluster_to_queries.par_iter().enumerate().for_each(|(c_idx, cluster_queries)| {
            if cluster_queries.is_empty() { return; }
            let start = offsets_sl[c_idx] as usize;
            let end = offsets_sl[c_idx+1] as usize;
            if start >= end { return; }

            for qchunk in cluster_queries.chunks(8) {
                // Thread-local heaps for this specific cluster scan (only 8 queries)
                let mut local_heaps: [BinaryHeap<Reverse<(u32, u32)>>; 8] = 
                    [(); 8].map(|_| BinaryHeap::with_capacity(top_k));

                // Pre-allocate LUTs for the current query chunk (8 queries)
                // These are reused across ALL clusters for this chunk
                let mut lut_sq = vec![0.0f32; d * num_levels * 8];
                let mut lut_qjl = vec![0.0f32; qjl_dim * 2 * 8];

                // PRE-COMPUTE LUT for the entire query chunk
                // SQ codes live in *rotated* residual space R(x-c); same as tq_batch_scan / tq_scan,
                // the SQ stage must use the rotated query (qjl_queries), not raw `queries`.
                // Phase 1 coarse routing still uses raw `queries` vs coarse centroids (IP, raw space).
                unsafe {
                    let mut centroid_bias = [0.0f32; 8];

                    for (lq, &(qi, ip)) in qchunk.iter().enumerate() {
                        let q_rot_row = qjl_queries_view.index_axis(Axis(0), qi);
                        let q_rot_sl = q_rot_row.as_slice().unwrap();

                        // Reuse coarse IP computed in Phase 1: ip = q · c_idx
                        centroid_bias[lq] = ip;
                        
                        for k in 0..d {
                            for l in 0..num_levels {
                                lut_sq[(k * num_levels + l) * 8 + lq] = cent_sl[l] * q_rot_sl[k];
                            }
                        }
                        for k in 0..qjl_dim {
                            // Stage 2 LUT: qjl_query * signs (+1 or -1)
                            lut_qjl[(k * 2 + 1) * 8 + lq] = q_rot_sl[k];  // index 1: positive sign
                            lut_qjl[(k * 2    ) * 8 + lq] = -q_rot_sl[k]; // index 0: negative sign
                        }
                    }

                    use std::arch::x86_64::*;
                    let v_centroid_bias = _mm256_loadu_ps(centroid_bias.as_ptr());
                    for i in start..end {
                        let rsq = i * packed_sq_d;
                        let rqj = i * (qjl_dim / 8);
                        let v_norms = _mm256_set1_ps(norms_sl[i]);
                        let v_res = _mm256_set1_ps(res_sl[i] * qjl_scale);

                        let mut v_sq = _mm256_setzero_ps();
                        if mse_bits == 1 {
                            for k in (0..d).step_by(8) {
                                let b = sq_flat[rsq + k / 8];
                                for bit in 0..8usize {
                                    let i_val = ((b >> bit) & 1) as usize;
                                    let v = _mm256_loadu_ps(lut_sq.as_ptr().add(((k + bit) * num_levels + i_val) * 8));
                                    v_sq = _mm256_add_ps(v_sq, v);
                                }
                            }
                        } else {
                            let bit_mask = (1 << mse_bits) - 1;
                            for k in (0..d).step_by(2) {
                                let b = sq_flat[rsq + k / 2];
                                let i0 = (b & bit_mask) as usize;
                                let i1 = ((b >> mse_bits) & bit_mask) as usize;
                                let v0 = _mm256_loadu_ps(lut_sq.as_ptr().add((k * num_levels + i0) * 8));
                                let v1 = _mm256_loadu_ps(lut_sq.as_ptr().add(((k+1) * num_levels + i1) * 8));
                                v_sq = _mm256_add_ps(v_sq, _mm256_add_ps(v0, v1));
                            }
                        }

                        let mut v_qjl = _mm256_setzero_ps();
                        for k in (0..qjl_dim).step_by(8) {
                            let b = signs_flat[rqj + k / 8];
                            for bit in 0..8usize {
                                let s = ((b >> bit) & 1) as usize;
                                let v = _mm256_loadu_ps(lut_qjl.as_ptr().add(((k+bit) * 2 + s) * 8));
                                v_qjl = _mm256_add_ps(v_qjl, v);
                            }
                        }

                        let v_residual = _mm256_mul_ps(_mm256_fmadd_ps(v_qjl, v_res, v_sq), v_norms);
                        let vf = _mm256_add_ps(v_residual, v_centroid_bias);
                        let mut tmp = [0.0f32; 8];
                        _mm256_storeu_ps(tmp.as_mut_ptr(), vf);

                        for (lq, &(qi, _ip)) in qchunk.iter().enumerate() {
                            let score_bits = float_to_ordered_u32(tmp[lq]);
                            let heap = &mut local_heaps[lq];
                            if heap.len() < top_k {
                                heap.push(Reverse((score_bits, i as u32)));
                            } else if score_bits > heap.peek().unwrap().0.0 {
                                heap.pop();
                                heap.push(Reverse((score_bits, i as u32)));
                            }
                        }
                    }
                }

                // Periodic Flush: Once per cluster scan for these 8 queries
                for (lq, &(qi, _ip)) in qchunk.iter().enumerate() {
                    let mut g_heap = global_heaps[qi].lock().unwrap();
                    let l_heap = &mut local_heaps[lq];
                    while let Some(Reverse(item)) = l_heap.pop() {
                        if g_heap.len() < top_k {
                            g_heap.push(Reverse(item));
                        } else if item.0 > g_heap.peek().unwrap().0.0 {
                            g_heap.pop();
                            g_heap.push(Reverse(item));
                        }
                    }
                }
            }
        });
    });

    // --- PHASE 4: EXTRACT TOP-K RESULTS ---
    let mut final_scores = vec![0.0f32; num_queries * top_k];
    let mut final_ids = vec![0i32; num_queries * top_k];

    for qi in 0..num_queries {
        let mut g_heap = global_heaps[qi].lock().unwrap();
        let mut results = Vec::new();
        while let Some(Reverse(entry)) = g_heap.pop() {
            results.push(entry);
        }
        results.reverse();
        
        for (k, (s_bits, g_id)) in results.into_iter().enumerate() {
            if k < top_k {
                final_scores[qi * top_k + k] = ordered_u32_to_float(s_bits);
                final_ids[qi * top_k + k] = g_id as i32;
            }
        }
    }

    let scores_arr = PyArray1::from_vec_bound(py, final_scores).reshape([num_queries, top_k])?;
    let ids_arr = PyArray1::from_vec_bound(py, final_ids).reshape([num_queries, top_k])?;

    Ok((scores_arr.unbind(), ids_arr.unbind()))
}
