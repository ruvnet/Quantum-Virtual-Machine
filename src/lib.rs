//! # Quantum Virtual Machine Scheduler
//!
//! A backend-agnostic quantum circuit scheduler with OpenQASM 3 support.
//! Provides circuit parsing, topology-aware scheduling, and composite circuit generation.

#![cfg_attr(not(feature = "std"), no_std)]

// WASM setup
#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn init_wasm() {
    #[cfg(feature = "wasm")]
    console_error_panic_hook::set_once();
}

// Core modules
pub mod error;
pub mod circuit_ir;
pub mod topology;
pub mod scheduler;
pub mod composer;

// API modules
pub mod api;

// WASM module
#[cfg(feature = "wasm")]
pub mod wasm;

// Re-exports for convenience
pub use error::{QvmError, Result};
pub use circuit_ir::{QuantumCircuit, Operation, Qubit, ClassicalBit, CircuitBuilder};
pub use topology::{Topology, Tile, Position, TopologyBuilder};
pub use scheduler::{Job, Schedule, Scheduler};
pub use composer::{CompositeCircuit, CircuitComposer};

/// QVM version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// High-level QVM scheduler interface
#[derive(Debug, Clone)]
pub struct QvmScheduler {
    topology: Topology,
    scheduler: Scheduler,
    composer: CircuitComposer,
}

impl QvmScheduler {
    /// Create a new QVM scheduler with the given topology
    pub fn new(topology: Topology) -> Self {
        Self {
            topology: topology.clone(),
            scheduler: Scheduler::new(topology.clone()),
            composer: CircuitComposer::new(topology),
        }
    }

    /// Schedule multiple circuits and generate composite output
    pub async fn schedule_circuits(
        &mut self,
        circuits: Vec<QuantumCircuit>,
    ) -> Result<CompositeCircuit> {
        let jobs: Vec<Job> = circuits
            .into_iter()
            .enumerate()
            .map(|(id, circuit)| Job::new(id, circuit))
            .collect();

        let schedule = self.scheduler.schedule(jobs).await?;
        self.composer.compose(schedule).await
    }

    /// Schedule multiple circuits (convenience method)
    pub async fn schedule(
        &self,
        circuits: &[QuantumCircuit],
    ) -> Result<CompositeCircuit> {
        let jobs: Vec<Job> = circuits
            .iter()
            .enumerate()
            .map(|(id, circuit)| Job::new(id, circuit.clone()))
            .collect();

        let mut scheduler = self.scheduler.clone();
        let schedule = scheduler.schedule(jobs).await?;
        
        let mut composer = self.composer.clone();
        composer.compose(schedule).await
    }

    /// Get the underlying topology
    pub fn topology(&self) -> &Topology {
        &self.topology
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;

    #[test]
    fn test_qvm_scheduler_creation() {
        let topology = TopologyBuilder::grid(3, 3);
        let scheduler = QvmScheduler::new(topology);
        assert_eq!(scheduler.topology().qubit_count(), 9);
    }

    #[cfg(feature = "std")]
    #[tokio::test]
    async fn test_empty_schedule() {
        let topology = TopologyBuilder::grid(2, 2);
        let mut scheduler = QvmScheduler::new(topology);
        
        let result = scheduler.schedule_circuits(vec![]).await;
        assert!(result.is_ok());
        
        let composite = result.unwrap();
        assert!(composite.circuits().is_empty());
    }
}
