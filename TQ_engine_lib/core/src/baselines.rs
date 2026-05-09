use numpy::{PyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub fn sq_scan(
    py: Python,
    query: PyReadonlyArray1<f32>,     // (D)
    codes: PyReadonlyArray2<u8>,      // (N, packed_D)
    _centroids: PyReadonlyArray1<f32>, // (256)
    norms: PyReadonlyArray1<f32>,     // (N)
    dim: usize,
    bits: usize,
) -> PyResult<PyObject> {
    let q_view = query.as_array();
    let codes_view = codes.as_array();
    let norms_view = norms.as_array();
    let n = codes_view.shape()[0];
    
    let mut output = vec![0.0f32; n];

    py.allow_threads(|| {
        output.par_iter_mut().enumerate().for_each(|(i, out_score)| {
            let q_ptr = q_view.as_ptr();
            let k_ptr = codes_view.row(i).as_ptr();
            let mut score_sum = 0.0f32;

            if bits == 8 {
                // SQ 8-bit (1 byte = 1 value)
                for d in 0..dim {
                    unsafe {
                        score_sum += (*q_ptr.add(d)) * (*k_ptr.add(d) as f32);
                    }
                }
            } else if bits == 4 {
                // SQ 4-bit (1 byte = 2 values)
                for d_packed in 0..(dim / 2) {
                    unsafe {
                        let byte = *k_ptr.add(d_packed);
                        let v1 = (byte >> 4) as f32;
                        let v2 = (byte & 0x0F) as f32;
                        score_sum += (*q_ptr.add(d_packed * 2)) * v1;
                        score_sum += (*q_ptr.add(d_packed * 2 + 1)) * v2;
                    }
                }
            } else if bits == 2 {
                // SQ 2-bit (1 byte = 4 values)
                for d_packed in 0..(dim / 4) {
                    unsafe {
                        let byte = *k_ptr.add(d_packed);
                        let v1 = (byte >> 6) as f32;
                        let v2 = ((byte >> 4) & 0x03) as f32;
                        let v3 = ((byte >> 2) & 0x03) as f32;
                        let v4 = (byte & 0x03) as f32;
                        score_sum += (*q_ptr.add(d_packed * 4)) * v1;
                        score_sum += (*q_ptr.add(d_packed * 4 + 1)) * v2;
                        score_sum += (*q_ptr.add(d_packed * 4 + 2)) * v3;
                        score_sum += (*q_ptr.add(d_packed * 4 + 3)) * v4;
                    }
                }
            }
            *out_score = score_sum * norms_view[i];
        });
    });

    let array = PyArray::from_vec_bound(py, output);
    Ok(array.into())
}

#[pyfunction]
pub fn pq_scan(
    py: Python,
    codes: PyReadonlyArray2<u8>,      // (N, M)
    dist_table: PyReadonlyArray2<f32>, // (M, 256)
) -> PyResult<PyObject> {
    let codes_view = codes.as_array();
    let dist_view = dist_table.as_array();
    let n = codes_view.shape()[0];
    let m = codes_view.shape()[1];
    
    let mut output = vec![0.0f32; n];

    py.allow_threads(|| {
        output.par_iter_mut().enumerate().for_each(|(i, out_score)| {
            let mut sum = 0.0f32;
            for j in 0..m {
                let code = codes_view[[i, j]] as usize;
                sum += dist_view[[j, code]];
            }
            *out_score = sum;
        });
    });

    let array = PyArray::from_vec_bound(py, output);
    Ok(array.into())
}
