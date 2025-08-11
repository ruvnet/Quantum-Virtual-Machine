//! Output generation for composite circuits

use crate::{QvmError, Result};
use crate::composer::{CompositeCircuit, ComposedCircuit, OutputFormat};
use serde::{Deserialize, Serialize};

/// Output generator for different formats
#[derive(Debug, Clone)]
pub struct OutputGenerator {
    config: OutputConfig,
}

/// Output generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output format
    pub format: OutputFormat,
    /// Include timing information
    pub include_timing: bool,
    /// Include mapping information
    pub include_mapping: bool,
    /// Include metadata comments
    pub include_metadata: bool,
    /// Optimization level for output
    pub optimization_level: u8,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::OpenQASM3,
            include_timing: true,
            include_mapping: true,
            include_metadata: true,
            optimization_level: 1,
        }
    }
}

impl OutputGenerator {
    /// Create a new output generator
    pub fn new() -> Self {
        Self {
            config: OutputConfig::default(),
        }
    }

    /// Create generator with custom configuration
    pub fn with_config(config: OutputConfig) -> Self {
        Self { config }
    }

    /// Generate QASM output for a composite circuit
    pub fn generate_qasm(&self, composite: &CompositeCircuit) -> Result<String> {
        match self.config.format {
            OutputFormat::OpenQASM3 => self.generate_qasm3(composite),
            OutputFormat::OpenQASM2 => self.generate_qasm2(composite),
            OutputFormat::Custom => self.generate_custom(composite),
        }
    }

    /// Generate OpenQASM 3 output
    fn generate_qasm3(&self, composite: &CompositeCircuit) -> Result<String> {
        let mut output = String::new();

        // Header
        output.push_str("OPENQASM 3.0;\n");
        output.push_str("include \"stdgates.inc\";\n");
        
        // Add timing support if needed
        if self.config.include_timing {
            output.push_str("include \"timing.inc\";\n");
        }
        output.push_str("\n");

        // Metadata comments
        if self.config.include_metadata {
            output.push_str(&format!("// Composite circuit with {} components\n", composite.circuit_count()));
            output.push_str(&format!("// Total duration: {} microseconds\n", composite.total_duration()));
            output.push_str(&format!("// Generated at: {}\n", composite.metadata.created_at));
            output.push_str("\n");
        }

        // Determine global resource requirements
        let (max_qubits, max_classical) = self.calculate_global_resources(composite);

        // Global declarations
        if max_qubits > 0 {
            output.push_str(&format!("qubit[{}] q;\n", max_qubits));
        }
        if max_classical > 0 {
            output.push_str(&format!("bit[{}] c;\n", max_classical));
        }
        output.push_str("\n");

        // Generate circuit content
        for (i, circuit) in composite.circuits().iter().enumerate() {
            output.push_str(&format!("// Circuit {} (Job {})\n", i, circuit.job_id));
            
            if self.config.include_timing {
                output.push_str(&format!("// Start time: {} μs, Duration: {} μs\n", 
                                       circuit.timing.start_time, circuit.timing.duration));
            }

            if self.config.include_mapping {
                output.push_str(&format!("// Qubit mapping: {:?}\n", circuit.mapping.qubit_mapping));
            }

            output.push_str(&self.generate_circuit_qasm3(circuit)?);
            output.push_str("\n");
        }

        Ok(output)
    }

    /// Generate OpenQASM 2 output (simplified)
    fn generate_qasm2(&self, composite: &CompositeCircuit) -> Result<String> {
        let mut output = String::new();

        // Header
        output.push_str("OPENQASM 2.0;\n");
        output.push_str("include \"qelib1.inc\";\n\n");

        // Calculate resources
        let (max_qubits, max_classical) = self.calculate_global_resources(composite);

        // Declarations
        if max_qubits > 0 {
            output.push_str(&format!("qreg q[{}];\n", max_qubits));
        }
        if max_classical > 0 {
            output.push_str(&format!("creg c[{}];\n", max_classical));
        }
        output.push_str("\n");

        // Generate circuits (simplified)
        for circuit in composite.circuits() {
            output.push_str(&format!("// Job {}\n", circuit.job_id));
            output.push_str(&self.generate_circuit_qasm2(circuit)?);
            output.push_str("\n");
        }

        Ok(output)
    }

    /// Generate custom format output
    fn generate_custom(&self, composite: &CompositeCircuit) -> Result<String> {
        // For now, delegate to OpenQASM 3
        self.generate_qasm3(composite)
    }

    /// Generate QASM 3 for a single circuit
    fn generate_circuit_qasm3(&self, circuit: &ComposedCircuit) -> Result<String> {
        let mut output = String::new();

        // Add timing barriers if requested
        if self.config.include_timing && circuit.timing.start_time > 0 {
            output.push_str("// Timing synchronization\n");
            output.push_str("barrier q;\n");
        }

        // Add timing directive if start time is specified
        if self.config.include_timing && circuit.timing.start_time > 0 {
            output.push_str(&format!("// Start at: {} us\n", circuit.timing.start_time));
            output.push_str(&format!("delay[{}us] q;\n", circuit.timing.start_time));
        }

        // Convert operations
        for operation in &circuit.circuit.operations {
            let qasm_line = operation.to_qasm();
            
            // Apply qubit mapping
            let mapped_line = self.apply_qubit_mapping(&qasm_line, &circuit.mapping.qubit_mapping)?;
            output.push_str(&mapped_line);
            output.push_str("\n");
        }

        // Add end timing barrier
        if self.config.include_timing {
            output.push_str("barrier q;\n");
        }

        Ok(output)
    }

    /// Generate QASM 2 for a single circuit
    fn generate_circuit_qasm2(&self, circuit: &ComposedCircuit) -> Result<String> {
        let mut output = String::new();

        // Convert operations (simplified)
        for operation in &circuit.circuit.operations {
            let qasm_line = operation.to_qasm();
            // Convert QASM 3 to QASM 2 syntax (basic conversion)
            let qasm2_line = self.convert_to_qasm2(&qasm_line)?;
            let mapped_line = self.apply_qubit_mapping(&qasm2_line, &circuit.mapping.qubit_mapping)?;
            output.push_str(&mapped_line);
            output.push_str("\n");
        }

        Ok(output)
    }

    /// Apply qubit mapping to a QASM line
    fn apply_qubit_mapping(&self, qasm_line: &str, mapping: &[usize]) -> Result<String> {
        let mut result = qasm_line.to_string();
        
        // Handle various QASM patterns
        // For single qubit operations: q[n]
        for (logical, &physical) in mapping.iter().enumerate() {
            let logical_pattern = format!("q[{}]", logical);
            let physical_replacement = format!("q[{}]", physical);
            result = result.replace(&logical_pattern, &physical_replacement);
        }
        
        // Handle multi-qubit gate patterns: q[i], q[j]
        for (logical, &physical) in mapping.iter().enumerate() {
            let logical_pattern = format!("q[{}],", logical);
            let physical_replacement = format!("q[{}],", physical);
            result = result.replace(&logical_pattern, &physical_replacement);
            
            let logical_pattern = format!(", q[{}]", logical);
            let physical_replacement = format!(", q[{}]", physical);
            result = result.replace(&logical_pattern, &physical_replacement);
        }

        Ok(result)
    }

    /// Convert QASM 3 line to QASM 2 (basic conversion)
    fn convert_to_qasm2(&self, qasm3_line: &str) -> Result<String> {
        // Basic conversions
        let mut result = qasm3_line.replace("bit[", "creg c");
        result = result.replace("qubit[", "qreg q");
        
        // Convert measurement syntax
        if result.contains("= measure") {
            // Convert "c[0] = measure q[0];" to "measure q[0] -> c[0];"
            result = result.replace("c[0] = measure q[0];", "measure q[0] -> c[0];");
        }

        Ok(result)
    }

    /// Calculate global resource requirements
    fn calculate_global_resources(&self, composite: &CompositeCircuit) -> (usize, usize) {
        let mut max_qubits = 0;
        let mut max_classical = 0;

        for circuit in composite.circuits() {
            if let Some(&max_physical_qubit) = circuit.mapping.qubit_mapping.iter().max() {
                max_qubits = max_qubits.max(max_physical_qubit + 1);
            }
            if let Some(&max_physical_classical) = circuit.mapping.classical_mapping.iter().max() {
                max_classical = max_classical.max(max_physical_classical + 1);
            }
        }

        (max_qubits, max_classical)
    }

    /// Generate summary statistics
    pub fn generate_summary(&self, composite: &CompositeCircuit) -> Result<String> {
        let mut summary = String::new();

        summary.push_str("# Composite Circuit Summary\n\n");
        summary.push_str(&format!("- Total circuits: {}\n", composite.circuit_count()));
        summary.push_str(&format!("- Total duration: {} μs\n", composite.total_duration()));
        summary.push_str(&format!("- Total qubits used: {}\n", composite.resource_summary.total_qubits_used));
        summary.push_str(&format!("- Peak qubit usage: {}\n", composite.resource_summary.peak_qubit_usage));
        summary.push_str(&format!("- Classical bits used: {}\n", composite.resource_summary.total_classical_used));
        summary.push_str(&format!("- Utilization efficiency: {:.2}%\n", 
                         composite.resource_summary.utilization_efficiency * 100.0));

        if composite.metadata.quality_metrics.expected_fidelity > 0.0 {
            summary.push_str(&format!("- Expected fidelity: {:.3}\n", 
                           composite.metadata.quality_metrics.expected_fidelity));
        }

        // Add timing information
        if composite.circuits().len() > 1 {
            let gaps: Vec<_> = composite.circuits().windows(2).map(|pair| {
                pair[1].timing.start_time.saturating_sub(pair[0].timing.start_time + pair[0].timing.duration)
            }).collect();
            
            if let Some(&min_gap) = gaps.iter().min() {
                summary.push_str(&format!("- Minimum gap between circuits: {} μs\n", min_gap));
            }
            
            if let Some(&max_gap) = gaps.iter().max() {
                summary.push_str(&format!("- Maximum gap between circuits: {} μs\n", max_gap));
            }
        }

        Ok(summary)
    }
}

impl Default for OutputGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composer::{CompositeCircuit, ComposedCircuit, CircuitTiming};
    use crate::composer::mapping::ResourceMapping;
    use crate::circuit_ir::{QuantumCircuit, Operation, SingleQubitGate, Qubit};
    use smallvec::SmallVec;

    #[test]
    fn test_output_generator_creation() {
        let generator = OutputGenerator::new();
        assert_eq!(generator.config.format, OutputFormat::OpenQASM3);
    }

    #[test]
    fn test_empty_composite_qasm() {
        let generator = OutputGenerator::new();
        let composite = CompositeCircuit::new();
        
        let qasm = generator.generate_qasm(&composite).unwrap();
        assert!(qasm.contains("OPENQASM 3.0"));
        assert!(qasm.contains("include \"stdgates.inc\""));
    }

    #[test]
    fn test_single_circuit_qasm() {
        let generator = OutputGenerator::new();
        let mut composite = CompositeCircuit::new();

        // Create a simple circuit
        let mut circuit = QuantumCircuit::new("test".to_string(), 2, 1);
        circuit.operations.push(Operation::SingleQubit {
            gate: SingleQubitGate::H,
            qubit: Qubit(0),
            parameters: SmallVec::new(),
        });

        let composed_circuit = ComposedCircuit {
            job_id: 0,
            circuit,
            timing: CircuitTiming {
                start_time: 0,
                duration: 1000,
                estimated_end_time: 1000,
            },
            mapping: ResourceMapping {
                qubit_mapping: vec![0, 1],
                classical_mapping: vec![0],
                metadata: Default::default(),
            },
        };

        composite.add_circuit(composed_circuit);
        
        let qasm = generator.generate_qasm(&composite).unwrap();
        assert!(qasm.contains("h q[0]"));
        assert!(qasm.contains("Job 0"));
    }

    #[test]
    fn test_qubit_mapping_application() {
        let generator = OutputGenerator::new();
        let mapping = vec![2, 5, 1]; // logical 0->physical 2, logical 1->physical 5, etc.
        
        let original = "h q[0]; cx q[0], q[1];";
        let mapped = generator.apply_qubit_mapping(original, &mapping).unwrap();
        
        assert!(mapped.contains("q[2]"));
        assert!(mapped.contains("q[5]"));
    }

    #[test]
    fn test_global_resources_calculation() {
        let generator = OutputGenerator::new();
        let mut composite = CompositeCircuit::new();

        let circuit1 = ComposedCircuit {
            job_id: 0,
            circuit: QuantumCircuit::new("test1".to_string(), 2, 1),
            timing: CircuitTiming { start_time: 0, duration: 1000, estimated_end_time: 1000 },
            mapping: ResourceMapping {
                qubit_mapping: vec![0, 1],
                classical_mapping: vec![0],
                metadata: Default::default(),
            },
        };

        let circuit2 = ComposedCircuit {
            job_id: 1,
            circuit: QuantumCircuit::new("test2".to_string(), 2, 1),
            timing: CircuitTiming { start_time: 1000, duration: 1000, estimated_end_time: 2000 },
            mapping: ResourceMapping {
                qubit_mapping: vec![3, 4],
                classical_mapping: vec![1],
                metadata: Default::default(),
            },
        };

        composite.add_circuit(circuit1);
        composite.add_circuit(circuit2);

        let (max_qubits, max_classical) = generator.calculate_global_resources(&composite);
        assert_eq!(max_qubits, 5); // qubit 4 is highest
        assert_eq!(max_classical, 2); // classical 1 is highest
    }
}