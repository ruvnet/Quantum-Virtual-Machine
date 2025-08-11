//! WASM bindings for the QVM scheduler

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;
use crate::{QvmScheduler, QuantumCircuit, Topology, TopologyBuilder, CircuitBuilder};

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub struct WasmScheduler {
    scheduler: QvmScheduler,
}

#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl WasmScheduler {
    /// Create a new scheduler with a grid topology
    #[wasm_bindgen(constructor)]
    pub fn new(rows: usize, cols: usize) -> Self {
        // Set panic hook for better error messages
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();
        
        let topology = TopologyBuilder::grid(rows, cols);
        Self {
            scheduler: QvmScheduler::new(topology),
        }
    }
    
    /// Create scheduler with linear topology
    #[wasm_bindgen(js_name = "fromLinear")]
    pub fn from_linear(size: usize) -> Self {
        let topology = TopologyBuilder::linear(size);
        Self {
            scheduler: QvmScheduler::new(topology),
        }
    }
    
    /// Create scheduler with ring topology
    #[wasm_bindgen(js_name = "fromRing")]
    pub fn from_ring(size: usize) -> Self {
        let topology = TopologyBuilder::ring(size);
        Self {
            scheduler: QvmScheduler::new(topology),
        }
    }
    
    /// Schedule multiple QASM circuits
    #[wasm_bindgen(js_name = "scheduleQasm")]
    pub async fn schedule_qasm(&self, qasm_strings: Vec<String>) -> Result<String, JsValue> {
        // Parse QASM strings into circuits
        let mut circuits = Vec::new();
        for (i, qasm) in qasm_strings.iter().enumerate() {
            match crate::circuit_ir::parse_qasm3(qasm) {
                Ok(circuit) => circuits.push(circuit),
                Err(e) => {
                    return Err(JsValue::from_str(&format!(
                        "Failed to parse circuit {}: {}", i, e
                    )));
                }
            }
        }
        
        // Schedule circuits
        match self.scheduler.schedule(&circuits).await {
            Ok(composite) => {
                // Generate output QASM
                composite.to_qasm()
                    .map_err(|e| JsValue::from_str(&format!("Failed to generate QASM: {}", e)))
            },
            Err(e) => Err(JsValue::from_str(&format!("Scheduling failed: {}", e))),
        }
    }
    
    /// Create and schedule demo circuits
    #[wasm_bindgen(js_name = "runDemo")]
    pub async fn run_demo(&self) -> Result<JsValue, JsValue> {
        // Create demo circuits
        let bell = CircuitBuilder::new("bell_state", 2, 2)
            .h(0).map_err(|e| JsValue::from_str(&e.to_string()))?
            .cx(0, 1).map_err(|e| JsValue::from_str(&e.to_string()))?
            .measure_all().map_err(|e| JsValue::from_str(&e.to_string()))?
            .build();
            
        let ghz = CircuitBuilder::new("ghz_state", 3, 3)
            .h(0).map_err(|e| JsValue::from_str(&e.to_string()))?
            .cx(0, 1).map_err(|e| JsValue::from_str(&e.to_string()))?
            .cx(1, 2).map_err(|e| JsValue::from_str(&e.to_string()))?
            .measure_all().map_err(|e| JsValue::from_str(&e.to_string()))?
            .build();
        
        let circuits = vec![bell, ghz];
        
        // Schedule
        match self.scheduler.schedule(&circuits).await {
            Ok(composite) => {
                // Create result object
                let result = serde_json::json!({
                    "success": true,
                    "circuits_scheduled": composite.circuits.len(),
                    "total_qubits": composite.total_qubits,
                    "total_cbits": composite.total_cbits,
                    "total_duration": composite.total_duration(),
                    "qasm": composite.to_qasm().unwrap_or_default(),
                });
                
                Ok(JsValue::from_str(&result.to_string()))
            },
            Err(e) => Err(JsValue::from_str(&format!("Scheduling failed: {}", e))),
        }
    }
}

/// Schedule QASM circuits with a given topology
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "scheduleWithTopology")]
pub async fn schedule_with_topology(
    qasm_strings: Vec<String>,
    topology_type: String,
    size: usize,
) -> Result<String, JsValue> {
    // Create topology based on type
    let topology = match topology_type.as_str() {
        "linear" => TopologyBuilder::linear(size),
        "ring" => TopologyBuilder::ring(size),
        "star" => TopologyBuilder::star(size),
        "grid" => {
            let sqrt = (size as f64).sqrt() as usize;
            TopologyBuilder::grid(sqrt, sqrt)
        },
        _ => return Err(JsValue::from_str(&format!("Unknown topology type: {}", topology_type))),
    };
    
    let scheduler = QvmScheduler::new(topology);
    
    // Parse circuits
    let mut circuits = Vec::new();
    for (i, qasm) in qasm_strings.iter().enumerate() {
        match crate::circuit_ir::parse_qasm3(qasm) {
            Ok(circuit) => circuits.push(circuit),
            Err(e) => {
                return Err(JsValue::from_str(&format!(
                    "Failed to parse circuit {}: {}", i, e
                )));
            }
        }
    }
    
    // Schedule
    match scheduler.schedule(&circuits).await {
        Ok(composite) => {
            composite.to_qasm()
                .map_err(|e| JsValue::from_str(&format!("Failed to generate QASM: {}", e)))
        },
        Err(e) => Err(JsValue::from_str(&format!("Scheduling failed: {}", e))),
    }
}

/// Get version information
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = "getVersion")]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}