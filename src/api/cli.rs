//! Command-line interface for the QVM scheduler

use crate::{QvmScheduler, QvmError, Result};
use crate::circuit_ir::QuantumCircuit;
use crate::topology::{Topology, TopologyBuilder};
use serde::{Deserialize, Serialize};

#[cfg(feature = "std")]
use std::path::Path;

/// CLI configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliConfig {
    /// Input QASM files
    pub input_files: Vec<String>,
    /// Output file path
    pub output_file: String,
    /// Topology configuration
    pub topology: TopologyConfig,
    /// Buffer zones between circuits
    pub buffer_zones: usize,
    /// Scheduling strategy
    pub strategy: SchedulingStrategy,
}

/// Topology configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopologyConfig {
    /// Grid topology with dimensions
    Grid { width: usize, height: usize },
    /// Linear topology with size
    Linear { size: usize },
    /// Custom topology from file
    Custom { file: String },
}

/// Available scheduling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingStrategy {
    /// First-fit decreasing
    FirstFitDecreasing,
    /// Best-fit
    BestFit,
    /// Worst-fit
    WorstFit,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            input_files: vec![],
            output_file: "output.qasm".to_string(),
            topology: TopologyConfig::Grid { width: 5, height: 5 },
            buffer_zones: 1,
            strategy: SchedulingStrategy::FirstFitDecreasing,
        }
    }
}

impl CliConfig {
    /// Create topology from configuration
    pub fn create_topology(&self) -> Result<Topology> {
        match &self.topology {
            TopologyConfig::Grid { width, height } => {
                Ok(TopologyBuilder::grid(*width, *height))
            }
            TopologyConfig::Linear { size } => {
                Ok(TopologyBuilder::linear(*size))
            }
            TopologyConfig::Custom { file: _ } => {
                // TODO: Implement custom topology loading
                Err(QvmError::config_error("Custom topologies not yet implemented"))
            }
        }
    }

    /// Load configuration from JSON file
    #[cfg(feature = "std")]
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: CliConfig = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to JSON file
    #[cfg(feature = "std")]
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

/// CLI runner for QVM operations
pub struct CliRunner {
    config: CliConfig,
    scheduler: QvmScheduler,
}

impl CliRunner {
    /// Create a new CLI runner with configuration
    pub fn new(config: CliConfig) -> Result<Self> {
        let topology = config.create_topology()?;
        let scheduler = QvmScheduler::new(topology);
        
        Ok(Self { config, scheduler })
    }

    /// Run the scheduling process
    #[cfg(feature = "std")]
    pub async fn run(&mut self) -> Result<()> {
        // Load input circuits
        let mut circuits = Vec::new();
        for input_file in &self.config.input_files {
            let circuit = self.load_circuit(input_file)?;
            circuits.push(circuit);
        }

        // Schedule circuits
        let composite = self.scheduler.schedule_circuits(circuits).await?;

        // Save output
        self.save_output(&composite).await?;

        println!("Successfully scheduled {} circuits", composite.circuits().len());
        println!("Output saved to: {}", self.config.output_file);

        Ok(())
    }

    #[cfg(feature = "std")]
    fn load_circuit(&self, file_path: &str) -> Result<QuantumCircuit> {
        let content = std::fs::read_to_string(file_path)?;
        QuantumCircuit::from_qasm(&content)
    }

    #[cfg(feature = "std")]
    async fn save_output(&self, composite: &crate::CompositeCircuit) -> Result<()> {
        let qasm_output = composite.to_qasm()?;
        std::fs::write(&self.config.output_file, qasm_output)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CliConfig::default();
        assert_eq!(config.buffer_zones, 1);
        assert!(config.input_files.is_empty());
    }

    #[test]
    fn test_topology_creation() {
        let config = CliConfig {
            topology: TopologyConfig::Grid { width: 3, height: 3 },
            ..Default::default()
        };

        let topology = config.create_topology().unwrap();
        assert_eq!(topology.qubit_count(), 9);
    }
}