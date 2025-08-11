//! Quantum Circuit Intermediate Representation
//! 
//! Provides data structures and parsing for quantum circuits with OpenQASM 3 support.

pub mod operations;
pub mod parser;
pub mod builder;
pub mod parser_demo;

pub use operations::*;
pub use parser::*;
pub use builder::*;

use crate::{QvmError, Result};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Quantum bit identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Qubit(pub usize);

impl Qubit {
    /// Create a new qubit with the given index
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Get the qubit index
    pub fn index(&self) -> usize {
        self.0
    }
}

impl From<usize> for Qubit {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

/// Classical bit identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ClassicalBit(pub usize);

impl ClassicalBit {
    /// Create a new classical bit with the given index
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Get the classical bit index
    pub fn index(&self) -> usize {
        self.0
    }
}

impl From<usize> for ClassicalBit {
    fn from(index: usize) -> Self {
        Self(index)
    }
}

/// Complete quantum circuit representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuit {
    /// Circuit name/identifier
    pub name: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of classical bits
    pub num_classical: usize,
    /// Operations in the circuit
    pub operations: Vec<Operation>,
    /// Circuit metadata
    pub metadata: CircuitMetadata,
}

/// Circuit metadata for scheduling and optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CircuitMetadata {
    /// Estimated execution time
    pub estimated_time: Option<f64>,
    /// Required qubit connectivity
    pub connectivity_requirements: Vec<(Qubit, Qubit)>,
    /// Classical readout requirements
    pub measurement_requirements: Vec<Qubit>,
    /// Priority for scheduling (higher = more priority)
    pub priority: i32,
    /// Custom tags
    pub tags: Vec<String>,
}

impl QuantumCircuit {
    /// Create a new empty quantum circuit
    pub fn new(name: String, num_qubits: usize, num_classical: usize) -> Self {
        Self {
            name,
            num_qubits,
            num_classical,
            operations: Vec::new(),
            metadata: CircuitMetadata::default(),
        }
    }

    /// Parse a circuit from OpenQASM 3 string
    pub fn from_qasm(qasm: &str) -> Result<Self> {
        parser::parse_qasm3(qasm)
    }

    /// Convert circuit to OpenQASM 3 string
    pub fn to_qasm(&self) -> Result<String> {
        let mut qasm = String::new();
        
        // Header
        qasm.push_str(&format!("// Circuit: {}\n", self.name));
        qasm.push_str("OPENQASM 3.0;\n");
        qasm.push_str("include \"stdgates.inc\";\n\n");

        // Qubit declarations
        if self.num_qubits > 0 {
            qasm.push_str(&format!("qubit[{}] q;\n", self.num_qubits));
        }

        // Classical bit declarations
        if self.num_classical > 0 {
            qasm.push_str(&format!("bit[{}] c;\n", self.num_classical));
        }

        if self.num_qubits > 0 || self.num_classical > 0 {
            qasm.push_str("\n");
        }

        // Operations
        for operation in &self.operations {
            qasm.push_str(&operation.to_qasm());
            qasm.push_str("\n");
        }

        Ok(qasm)
    }

    /// Add an operation to the circuit
    pub fn add_operation(&mut self, operation: Operation) {
        // Validate operation against circuit dimensions
        self.validate_operation(&operation).expect("Invalid operation for circuit");
        self.operations.push(operation);
    }

    /// Validate an operation against circuit constraints
    pub fn validate_operation(&self, operation: &Operation) -> Result<()> {
        match operation {
            Operation::SingleQubit { qubit, .. } => {
                if qubit.index() >= self.num_qubits {
                    return Err(QvmError::invalid_circuit(
                        format!("Qubit {} out of range for circuit with {} qubits", 
                               qubit.index(), self.num_qubits)
                    ));
                }
            }
            Operation::TwoQubit { control, target, .. } => {
                if control.index() >= self.num_qubits || target.index() >= self.num_qubits {
                    return Err(QvmError::invalid_circuit("Qubit out of range".to_string()));
                }
                if control == target {
                    return Err(QvmError::invalid_circuit("Control and target cannot be the same".to_string()));
                }
            }
            Operation::Measurement { qubit, classical, .. } => {
                if qubit.index() >= self.num_qubits {
                    return Err(QvmError::invalid_circuit("Measurement qubit out of range".to_string()));
                }
                if classical.index() >= self.num_classical {
                    return Err(QvmError::invalid_circuit("Classical bit out of range".to_string()));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Get all qubits used in the circuit
    pub fn used_qubits(&self) -> Vec<Qubit> {
        let mut used = std::collections::HashSet::new();
        for op in &self.operations {
            op.qubits().into_iter().for_each(|q| { used.insert(q); });
        }
        let mut result: Vec<_> = used.into_iter().collect();
        result.sort_by_key(|q| q.index());
        result
    }

    /// Get circuit depth (number of time steps)
    pub fn depth(&self) -> usize {
        // Simple depth calculation - can be improved with dependency analysis
        self.operations.len()
    }

    /// Get two-qubit operation count
    pub fn two_qubit_gate_count(&self) -> usize {
        self.operations.iter()
            .filter(|op| matches!(op, Operation::TwoQubit { .. }))
            .count()
    }

    /// Check if circuit uses the given qubit
    pub fn uses_qubit(&self, qubit: Qubit) -> bool {
        self.operations.iter()
            .any(|op| op.qubits().contains(&qubit))
    }

    /// Get all two-qubit interactions in the circuit
    pub fn two_qubit_interactions(&self) -> Vec<(Qubit, Qubit)> {
        self.operations.iter()
            .filter_map(|op| {
                if let Operation::TwoQubit { control, target, .. } = op {
                    Some((*control, *target))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Default for QuantumCircuit {
    fn default() -> Self {
        Self::new("circuit".to_string(), 0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_creation() {
        let circuit = QuantumCircuit::new("test".to_string(), 3, 2);
        assert_eq!(circuit.name, "test");
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.num_classical, 2);
    }

    #[test]
    fn test_qubit_validation() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2, 1);
        
        // Valid operation
        let valid_op = Operation::SingleQubit {
            gate: SingleQubitGate::X,
            qubit: Qubit(0),
            parameters: SmallVec::new(),
        };
        circuit.add_operation(valid_op);
        assert_eq!(circuit.operations.len(), 1);

        // Invalid qubit index should fail validation
        let invalid_op = Operation::SingleQubit {
            gate: SingleQubitGate::X,
            qubit: Qubit(5),
            parameters: SmallVec::new(),
        };
        assert!(circuit.validate_operation(&invalid_op).is_err());
    }

    #[test]
    fn test_qasm_generation() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2, 1);
        circuit.add_operation(Operation::SingleQubit {
            gate: SingleQubitGate::X,
            qubit: Qubit(0),
            parameters: SmallVec::new(),
        });

        let qasm = circuit.to_qasm().unwrap();
        assert!(qasm.contains("OPENQASM 3.0"));
        assert!(qasm.contains("qubit[2] q"));
        assert!(qasm.contains("x q[0]"));
    }

    #[test]
    fn test_used_qubits() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 5, 0);
        circuit.add_operation(Operation::SingleQubit {
            gate: SingleQubitGate::X,
            qubit: Qubit(0),
            parameters: SmallVec::new(),
        });
        circuit.add_operation(Operation::SingleQubit {
            gate: SingleQubitGate::H,
            qubit: Qubit(3),
            parameters: SmallVec::new(),
        });

        let used = circuit.used_qubits();
        assert_eq!(used, vec![Qubit(0), Qubit(3)]);
    }
}