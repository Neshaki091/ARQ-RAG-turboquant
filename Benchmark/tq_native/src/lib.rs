use pyo3::prelude::*;

mod turboquant;
mod sq8;
mod pq;

#[pymodule]
fn tq_native_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(turboquant::mse_score_simd, m)?)?;
    m.add_function(wrap_pyfunction!(turboquant::qjl_score_simd, m)?)?;
    m.add_function(wrap_pyfunction!(sq8::sq8_score_simd, m)?)?;
    m.add_function(wrap_pyfunction!(pq::pq_score_simd, m)?)?;
    Ok(())
}
