use numpy::{PyArray, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[pyfunction]
pub fn sq8_score_simd(
    py: Python,
    query: PyReadonlyArray2<f32>,     // (BH, D)
    keys_u8: PyReadonlyArray3<u8>,    // (BH, N, D)
) -> PyResult<PyObject> {
    let query_view = query.as_array();
    let bh = query_view.shape()[0];
    let d = query_view.shape()[1];
    
    let keys_view = keys_u8.as_array();
    let n = keys_view.shape()[1];
    
    let mut output = vec![0.0f32; bh * n];

    py.allow_threads(|| {
        const CHUNK_SIZE: usize = 1000;
        output.par_chunks_mut(CHUNK_SIZE).enumerate().for_each(|(i_chunk, chunk_output)| {
            let start_idx = i_chunk * CHUNK_SIZE;
            for sub_idx in 0..chunk_output.len() {
                let global_idx = start_idx + sub_idx;
                let i_bh = global_idx / n;
                let i_n = global_idx % n;
                
                let q_ptr = query_view.row(i_bh).as_ptr();
                let k_ptr = keys_view.slice(ndarray::s![i_bh, i_n, ..]).as_ptr();
                
                let mut score_sum = 0.0f32;
                let mut i_d = 0;

                #[cfg(target_arch = "x86_64")]
                unsafe {
                    let mut v_acc = _mm256_setzero_ps();
                    
                    while i_d + 8 <= d {
                        let v_q = _mm256_loadu_ps(q_ptr.add(i_d));
                        let k_bytes = *(k_ptr.add(i_d) as *const u64);
                        let v_k_i32 = _mm256_cvtepu8_epi32(_mm_set_epi64x(0, k_bytes as i64));
                        let v_k_f32 = _mm256_cvtepi32_ps(v_k_i32);
                        v_acc = _mm256_fmadd_ps(v_q, v_k_f32, v_acc);
                        i_d += 8;
                    }
                    
                    let mut tmp = [0.0f32; 8];
                    _mm256_storeu_ps(tmp.as_mut_ptr(), v_acc);
                    score_sum = tmp.iter().sum();
                }

                while i_d < d {
                    unsafe {
                        score_sum += (*q_ptr.add(i_d)) * (*k_ptr.add(i_d) as f32);
                    }
                    i_d += 1;
                }
                chunk_output[sub_idx] = score_sum;
            }
        });
    });

    let array = PyArray::from_vec_bound(py, output).reshape((bh, n))?;
    Ok(array.into())
}
