use pyo3::prelude::*;

mod turboquant;
mod sq8;
mod pq;

#[pymodule]
fn tq_native_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Tự động sử dụng số luồng tối đa của hệ thống
    let _ = rayon::ThreadPoolBuilder::new().build_global();

    // NEW CORE: Scalar Quantization (b-1) + QJL (1)
    m.add_function(wrap_pyfunction!(turboquant::tq_scan, m)?)?;
    
    // Legacy functions
    m.add_function(wrap_pyfunction!(turboquant::tq_master_scan, m)?)?;
    m.add_function(wrap_pyfunction!(turboquant::mse_score_simd, m)?)?;
    
    // Baseline comparisons
    m.add_function(wrap_pyfunction!(sq8::sq8_score_simd, m)?)?;
    m.add_function(wrap_pyfunction!(pq::pq_score_simd, m)?)?;
    Ok(())
}
