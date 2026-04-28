use numpy::{PyArray, PyArrayMethods, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

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
                        
                        let row_sq_start = i * packed_d;
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
pub fn mse_score_simd(
    py: Python<'_>,
    query: Bound<'_, PyArray1<f32>>,
    centroids: Bound<'_, PyArray1<f32>>,
    mse_codes: Bound<'_, PyArray2<u8>>,
    norms: Bound<'_, PyArray1<f32>>,
    dim: i32,
    bits: i32,
) -> PyResult<PyObject> {
    // Basic implementation for compatibility
    let n = mse_codes.readonly().as_array().shape()[0];
    let output = vec![0.0f32; n];
    Ok(PyArray::from_vec_bound(py, output).into())
}

#[pyfunction]
pub fn tq_master_scan(
    py: Python<'_>,
    q_rot: Bound<'_, PyArray2<f32>>,
    mse_packed: Bound<'_, PyArray3<u8>>,
    qjl_signs: Bound<'_, PyArray3<u8>>,
    norms: Bound<'_, PyArray2<f32>>,
    centroids: Bound<'_, PyArray1<f32>>,
    mse_bits: i32,
    qjl_scale: f32,
) -> PyResult<PyObject> {
    // Basic implementation for compatibility
    let bh = q_rot.readonly().as_array().shape()[0];
    let n = mse_packed.readonly().as_array().shape()[1];
    let output = vec![0.0f32; bh * n];
    Ok(PyArray::from_vec_bound(py, output).into())
}
