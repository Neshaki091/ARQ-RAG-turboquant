use numpy::{PyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn mse_score_simd(
    py: Python,
    query_rot: PyReadonlyArray2<f32>,     // (BH, D)
    mse_packed: PyReadonlyArray3<u8>,    // (BH, N, PACKED_D)
    norms: PyReadonlyArray2<f32>,        // (BH, N)
    centroids: PyReadonlyArray1<f32>,    // (N_CLUSTERS)
    mse_bits: i32,
) -> PyResult<PyObject> {
    let query_rot_view = query_rot.as_array();
    let bh = query_rot_view.shape()[0];
    let d = query_rot_view.shape()[1];
    
    let mse_packed_view = mse_packed.as_array();
    let n = mse_packed_view.shape()[1];
    let packed_d = mse_packed_view.shape()[2];
    
    let norms_view = norms.as_array();
    let centroids_view = centroids.as_array();
    
    let mut output = vec![0.0f32; bh * n];
    
    let bits = mse_bits as u32;
    let vals_per_byte = match bits {
        1 => 8,
        2 => 4,
        3 | 4 => 2,
        _ => 1,
    };
    let bit_mask = (1 << bits) - 1;

    py.allow_threads(|| {
        const CHUNK_SIZE: usize = 10000;
        output.par_chunks_mut(CHUNK_SIZE).enumerate().for_each(|(i_chunk, chunk_output)| {
            let start_idx = i_chunk * CHUNK_SIZE;
            for sub_idx in 0..chunk_output.len() {
                let global_idx = start_idx + sub_idx;
                let i_bh = global_idx / n;
                let i_n = global_idx % n;
                
                let q = query_rot_view.row(i_bh);
                let mut score: f32 = 0.0;
                for i_p in 0..packed_d {
                    let packed = mse_packed_view[[i_bh, i_n, i_p]];
                    for sub in 0..vals_per_byte {
                        let coord_idx = i_p * vals_per_byte + sub;
                        if coord_idx < d {
                            let idx = ((packed >> (sub * bits as usize)) & bit_mask) as usize;
                            score += q[coord_idx] * centroids_view[idx];
                        }
                    }
                }
                chunk_output[sub_idx] = score * norms_view[[i_bh, i_n]];
            }
        });
    });

    let array = PyArray::from_vec_bound(py, output).reshape((bh, n))?;
    Ok(array.into())
}

#[pyfunction]
pub fn qjl_score_simd(
    py: Python,
    q_sketch: PyReadonlyArray2<f32>,      // (BH, D)
    qjl_signs: PyReadonlyArray3<u8>,     // (BH, N, PACKED_D_SIGNS)
    res_norms: PyReadonlyArray2<f32>,    // (BH, N)
    qjl_scale: f32,
) -> PyResult<PyObject> {
    let q_sketch_view = q_sketch.as_array();
    let bh = q_sketch_view.shape()[0];
    let d = q_sketch_view.shape()[1];
    
    let qjl_signs_view = qjl_signs.as_array();
    let n = qjl_signs_view.shape()[1];
    let packed_d_signs = qjl_signs_view.shape()[2];
    
    let res_norms_view = res_norms.as_array();
    
    let mut output = vec![0.0f32; bh * n];

    py.allow_threads(|| {
        const CHUNK_SIZE: usize = 10000;
        output.par_chunks_mut(CHUNK_SIZE).enumerate().for_each(|(i_chunk, chunk_output)| {
            let start_idx = i_chunk * CHUNK_SIZE;
            for sub_idx in 0..chunk_output.len() {
                let global_idx = start_idx + sub_idx;
                let i_bh = global_idx / n;
                let i_n = global_idx % n;
                
                let qs = q_sketch_view.row(i_bh);
                let mut dot: f32 = 0.0;
                for i_p in 0..packed_d_signs {
                    let packed = qjl_signs_view[[i_bh, i_n, i_p]];
                    for bit in 0..8 {
                        let coord_idx = i_p * 8 + bit;
                        if coord_idx < d {
                            let is_on = (packed >> bit) & 1;
                            let sign_val = if is_on == 1 { 1.0 } else { -1.0 };
                            dot += qs[coord_idx] * sign_val;
                        }
                    }
                }
                chunk_output[sub_idx] = dot * res_norms_view[[i_bh, i_n]] * qjl_scale;
            }
        });
    });

    let array = PyArray::from_vec_bound(py, output).reshape((bh, n))?;
    Ok(array.into())
}
