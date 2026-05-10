use pyo3::prelude::*;

mod turboquant;
mod sq8;
mod pq;
mod baselines;

#[pymodule]
fn tq_native_lib(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Threading:
    // - Default: use OS-reported available parallelism
    // - Override: set env var TQ_RAYON_THREADS (e.g. "8")
    // NOTE: build_global() can only be called once; ignore error if already initialized.
    let threads = 2* std::env::var("TQ_RAYON_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(0);
    let mut b = rayon::ThreadPoolBuilder::new();
    if threads > 0 {
        b = b.num_threads(threads);
    }
    let _ = b.build_global();

    // NEW CORE: Scalar Quantization (b-1) + QJL (1)
    m.add_function(wrap_pyfunction!(turboquant::tq_scan, m)?)?;
    m.add_function(wrap_pyfunction!(turboquant::tq_batch_scan, m)?)?;
    m.add_function(wrap_pyfunction!(turboquant::tq_ivf_online_scan, m)?)?;
    m.add_function(wrap_pyfunction!(turboquant::tq_quantize_rotated, m)?)?;
    
    // Legacy functions
    m.add_function(wrap_pyfunction!(turboquant::tq_master_scan, m)?)?;
    m.add_function(wrap_pyfunction!(turboquant::mse_score_simd, m)?)?;
    
    // Baseline comparisons (New names for stress_5m.py)
    m.add_function(wrap_pyfunction!(baselines::sq_scan, m)?)?;
    m.add_function(wrap_pyfunction!(baselines::pq_scan, m)?)?;

    // Original names
    m.add_function(wrap_pyfunction!(sq8::sq8_score_simd, m)?)?;
    m.add_function(wrap_pyfunction!(pq::pq_score_simd, m)?)?;
    
    Ok(())
}
