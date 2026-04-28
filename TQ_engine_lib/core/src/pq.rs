use numpy::{PyArray, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[pyfunction]
pub fn pq_score_simd(
    py: Python,
    _query: PyReadonlyArray2<f32>,         // (BH, D)
    codes: PyReadonlyArray3<u8>,         // (BH, N, M)
    precomputed_dist: PyReadonlyArray3<f32>, // (BH, M, 256)
) -> PyResult<PyObject> {
    let dist_view = precomputed_dist.as_array();
    let bh = dist_view.shape()[0];
    let m = dist_view.shape()[1];
    
    let codes_view = codes.as_array();
    let n = codes_view.shape()[1];
    
    let mut output = vec![0.0f32; bh * n];

    py.allow_threads(|| {
        output.par_chunks_mut(n).enumerate().for_each(|(i_bh, row_output)| {
            for i_m in 0..m {
                let lut = dist_view.slice(ndarray::s![i_bh, i_m, ..]);
                let lut_ptr = lut.as_ptr();

                let mut i_n = 0;
                
                #[cfg(target_arch = "x86_64")]
                {
                    while i_n + 8 <= n {
                        unsafe {
                            let mut indices = [0i32; 8];
                            for j in 0..8 {
                                indices[j] = codes_view[[i_bh, i_n + j, i_m]] as i32;
                            }

                            let v_indices = _mm256_loadu_si256(indices.as_ptr() as *const __m256i);
                            let v_dists = _mm256_i32gather_ps(lut_ptr, v_indices, 4);
                            let v_current = _mm256_loadu_ps(row_output.as_ptr().add(i_n));
                            let v_new = _mm256_add_ps(v_current, v_dists);
                            _mm256_storeu_ps(row_output.as_mut_ptr().add(i_n), v_new);
                        }
                        i_n += 8;
                    }
                }
                while i_n < n {
                    let code = codes_view[[i_bh, i_n, i_m]] as usize;
                    row_output[i_n] += lut[code];
                    i_n += 1;
                }
            }
        });
    });

    let array = PyArray::from_vec_bound(py, output).reshape((bh, n))?;
    Ok(array.into())
}
