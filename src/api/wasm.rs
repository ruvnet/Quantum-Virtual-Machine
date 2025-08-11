//! WebAssembly bindings for the QVM scheduler

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use crate::{QvmScheduler, QvmError, Result};
#[cfg(feature = "wasm")]
use crate::circuit_ir::QuantumCircuit;
#[cfg(feature = "wasm")]
use crate::topology::TopologyBuilder;

/// WASM-compatible scheduler interface
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmQvmScheduler {
    inner: QvmScheduler,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmQvmScheduler {
    /// Create a new scheduler with grid topology
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize) -> Self {
        let topology = TopologyBuilder::grid(width, height).build();
        let scheduler = QvmScheduler::new(topology);
        
        Self { inner: scheduler }
    }

    /// Create a scheduler with linear topology
    #[wasm_bindgen(js_name = "newLinear")]
    pub fn new_linear(size: usize) -> Self {
        let topology = TopologyBuilder::linear(size).build();
        let scheduler = QvmScheduler::new(topology);
        
        Self { inner: scheduler }
    }

    /// Schedule circuits from QASM strings
    #[wasm_bindgen(js_name = "scheduleQasm")]
    pub async fn schedule_qasm(&mut self, qasm_circuits: Vec<String>) -> Result<String, JsValue> {
        let circuits: Result<Vec<QuantumCircuit>, QvmError> = qasm_circuits
            .into_iter()
            .map(|qasm| QuantumCircuit::from_qasm(&qasm))
            .collect();

        let circuits = circuits.map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        let composite = self.inner
            .schedule_circuits(circuits)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let output = composite
            .to_qasm()
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(output)
    }

    /// Get topology information as JSON
    #[wasm_bindgen(js_name = "getTopologyInfo")]
    pub fn get_topology_info(&self) -> String {
        let topology = self.inner.topology();
        serde_json::to_string(&TopologyInfo {
            qubit_count: topology.qubit_count(),
            connectivity: topology.connectivity_degree(),
        }).unwrap_or_else(|_| "{}".to_string())
    }

    /// Get scheduler statistics
    #[wasm_bindgen(js_name = "getStats")]
    pub fn get_stats(&self) -> String {
        // TODO: Implement scheduler statistics
        "{}" .to_string()
    }
}

#[cfg(feature = "wasm")]
#[derive(serde::Serialize)]
struct TopologyInfo {
    qubit_count: usize,
    connectivity: f64,
}

/// Initialize the WASM module
#[cfg(feature = "wasm")]
#[wasm_bindgen(start)]
pub fn main() {
    crate::init_wasm();
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// Console log macro for WASM
#[cfg(feature = "wasm")]
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => ($crate::api::wasm::log(&format_args!($($t)*).to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "wasm")]
    #[test]
    fn test_wasm_scheduler_creation() {
        let scheduler = WasmQvmScheduler::new(3, 3);
        let info = scheduler.get_topology_info();
        assert!(info.contains("qubit_count"));
    }
}