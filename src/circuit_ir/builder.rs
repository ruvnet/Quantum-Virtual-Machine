//! Circuit builder for programmatic circuit construction

use crate::{QvmError, Result};
use crate::circuit_ir::{QuantumCircuit, Operation, SingleQubitGate, TwoQubitGate, Qubit, ClassicalBit, CircuitMetadata};
use smallvec::SmallVec;
use std::f64::consts::PI;

/// Builder for constructing quantum circuits programmatically
#[derive(Debug, Clone)]
pub struct CircuitBuilder {
    name: String,
    num_qubits: usize,
    num_classical: usize,
    operations: Vec<Operation>,
    metadata: CircuitMetadata,
}

impl CircuitBuilder {
    /// Create a new circuit builder
    pub fn new(name: impl Into<String>, num_qubits: usize, num_classical: usize) -> Self {
        Self {
            name: name.into(),
            num_qubits,
            num_classical,
            operations: Vec::new(),
            metadata: CircuitMetadata::default(),
        }
    }

    /// Build the final circuit
    pub fn build(self) -> QuantumCircuit {
        QuantumCircuit {
            name: self.name,
            num_qubits: self.num_qubits,
            num_classical: self.num_classical,
            operations: self.operations,
            metadata: self.metadata,
        }
    }

    /// Set circuit metadata
    pub fn with_metadata(mut self, metadata: CircuitMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set circuit priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.metadata.priority = priority;
        self
    }

    /// Add a tag to the circuit
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.metadata.tags.push(tag.into());
        self
    }

    // === Single-qubit gates ===

    /// Add an X (NOT) gate
    pub fn x(mut self, qubit: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::X,
            qubit: qubit.into(),
            parameters: SmallVec::new(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add a Y gate
    pub fn y(mut self, qubit: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::Y,
            qubit: qubit.into(),
            parameters: SmallVec::new(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add a Z gate
    pub fn z(mut self, qubit: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::Z,
            qubit: qubit.into(),
            parameters: SmallVec::new(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add a Hadamard gate
    pub fn h(mut self, qubit: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::H,
            qubit: qubit.into(),
            parameters: SmallVec::new(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add an S gate
    pub fn s(mut self, qubit: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::S,
            qubit: qubit.into(),
            parameters: SmallVec::new(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add an S-dagger gate
    pub fn sdg(mut self, qubit: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::Sdg,
            qubit: qubit.into(),
            parameters: SmallVec::new(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add a T gate
    pub fn t(mut self, qubit: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::T,
            qubit: qubit.into(),
            parameters: SmallVec::new(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add a T-dagger gate
    pub fn tdg(mut self, qubit: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::Tdg,
            qubit: qubit.into(),
            parameters: SmallVec::new(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add an RX rotation gate
    pub fn rx(mut self, angle: f64, qubit: impl Into<Qubit>) -> Result<Self> {
        let mut parameters = SmallVec::new();
        parameters.push(angle);
        
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::RX,
            qubit: qubit.into(),
            parameters,
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add an RY rotation gate
    pub fn ry(mut self, angle: f64, qubit: impl Into<Qubit>) -> Result<Self> {
        let mut parameters = SmallVec::new();
        parameters.push(angle);
        
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::RY,
            qubit: qubit.into(),
            parameters,
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add an RZ rotation gate
    pub fn rz(mut self, angle: f64, qubit: impl Into<Qubit>) -> Result<Self> {
        let mut parameters = SmallVec::new();
        parameters.push(angle);
        
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::RZ,
            qubit: qubit.into(),
            parameters,
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add a phase gate
    pub fn p(mut self, angle: f64, qubit: impl Into<Qubit>) -> Result<Self> {
        let mut parameters = SmallVec::new();
        parameters.push(angle);
        
        let operation = Operation::SingleQubit {
            gate: SingleQubitGate::P,
            qubit: qubit.into(),
            parameters,
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    // === Two-qubit gates ===

    /// Add a CNOT gate
    pub fn cx(mut self, control: impl Into<Qubit>, target: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::TwoQubit {
            gate: TwoQubitGate::CNOT,
            control: control.into(),
            target: target.into(),
            parameters: SmallVec::new(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add a CZ gate
    pub fn cz(mut self, control: impl Into<Qubit>, target: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::TwoQubit {
            gate: TwoQubitGate::CZ,
            control: control.into(),
            target: target.into(),
            parameters: SmallVec::new(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add a SWAP gate
    pub fn swap(mut self, qubit1: impl Into<Qubit>, qubit2: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::TwoQubit {
            gate: TwoQubitGate::SWAP,
            control: qubit1.into(),
            target: qubit2.into(),
            parameters: SmallVec::new(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add a controlled-phase gate
    pub fn cp(mut self, angle: f64, control: impl Into<Qubit>, target: impl Into<Qubit>) -> Result<Self> {
        let mut parameters = SmallVec::new();
        parameters.push(angle);
        
        let operation = Operation::TwoQubit {
            gate: TwoQubitGate::CP,
            control: control.into(),
            target: target.into(),
            parameters,
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    // === Measurement and control ===

    /// Add a measurement operation
    pub fn measure(mut self, qubit: impl Into<Qubit>, classical: impl Into<ClassicalBit>) -> Result<Self> {
        let operation = Operation::Measurement {
            qubit: qubit.into(),
            classical: classical.into(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add measurements for all qubits
    pub fn measure_all(mut self) -> Result<Self> {
        let min_bits = self.num_qubits.min(self.num_classical);
        for i in 0..min_bits {
            let operation = Operation::Measurement {
                qubit: Qubit(i),
                classical: ClassicalBit(i),
            };
            self.operations.push(operation);
        }
        Ok(self)
    }

    /// Add a reset operation
    pub fn reset(mut self, qubit: impl Into<Qubit>) -> Result<Self> {
        let operation = Operation::Reset {
            qubit: qubit.into(),
        };
        self.validate_operation(&operation)?;
        self.operations.push(operation);
        Ok(self)
    }

    /// Add a barrier
    pub fn barrier(mut self, qubits: &[Qubit]) -> Result<Self> {
        let operation = Operation::Barrier {
            qubits: qubits.iter().cloned().collect(),
        };
        // Barriers don't need validation against circuit constraints
        self.operations.push(operation);
        Ok(self)
    }

    /// Add a barrier across all qubits
    pub fn barrier_all(mut self) -> Result<Self> {
        let qubits: Vec<Qubit> = (0..self.num_qubits).map(Qubit).collect();
        self.barrier(&qubits)
    }

    // === Higher-level operations ===

    /// Create a Bell state preparation circuit
    pub fn bell_state(mut self, qubit1: impl Into<Qubit>, qubit2: impl Into<Qubit>) -> Result<Self> {
        let q1 = qubit1.into();
        let q2 = qubit2.into();
        self = self.h(q1)?.cx(q1, q2)?;
        Ok(self)
    }

    /// Create a GHZ state preparation circuit
    pub fn ghz_state(mut self, qubits: &[Qubit]) -> Result<Self> {
        if qubits.is_empty() {
            return Ok(self);
        }
        
        // Apply H to first qubit
        self = self.h(qubits[0])?;
        
        // Apply CNOT from first to all others
        for &target in &qubits[1..] {
            self = self.cx(qubits[0], target)?;
        }
        
        Ok(self)
    }

    /// Add a quantum Fourier transform
    pub fn qft(mut self, qubits: &[Qubit]) -> Result<Self> {
        let n = qubits.len();
        
        for i in 0..n {
            // Apply H gate
            self = self.h(qubits[i])?;
            
            // Apply controlled phase gates
            for j in (i + 1)..n {
                let angle = PI / (1 << (j - i)) as f64;
                self = self.cp(angle, qubits[j], qubits[i])?;
            }
        }
        
        // Reverse qubit order with SWAP gates
        for i in 0..(n / 2) {
            self = self.swap(qubits[i], qubits[n - 1 - i])?;
        }
        
        Ok(self)
    }

    // === Validation ===

    /// Validate an operation against circuit constraints
    fn validate_operation(&self, operation: &Operation) -> Result<()> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_circuit_building() {
        let circuit = CircuitBuilder::new("test", 2, 2)
            .h(0).unwrap()
            .cx(0, 1).unwrap()
            .measure_all().unwrap()
            .build();

        assert_eq!(circuit.name, "test");
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.operations.len(), 4); // H, CX, measure(0), measure(1)
    }

    #[test]
    fn test_bell_state() {
        let circuit = CircuitBuilder::new("bell", 2, 2)
            .bell_state(0, 1).unwrap()
            .build();

        assert_eq!(circuit.operations.len(), 2); // H, CX
        
        // Check first operation is H
        match &circuit.operations[0] {
            Operation::SingleQubit { gate: SingleQubitGate::H, qubit, .. } => {
                assert_eq!(*qubit, Qubit(0));
            }
            _ => panic!("Expected H gate"),
        }
    }

    #[test]
    fn test_validation() {
        let result = CircuitBuilder::new("test", 1, 1)
            .x(5); // Qubit 5 doesn't exist
        
        assert!(result.is_err());
    }

    #[test]
    fn test_qft() {
        let qubits = [Qubit(0), Qubit(1), Qubit(2)];
        let circuit = CircuitBuilder::new("qft", 3, 0)
            .qft(&qubits).unwrap()
            .build();

        // Should have H gates, controlled phase gates, and SWAP gates
        assert!(!circuit.operations.is_empty());
    }

    #[test]
    fn test_ghz_state() {
        let qubits = [Qubit(0), Qubit(1), Qubit(2)];
        let circuit = CircuitBuilder::new("ghz", 3, 0)
            .ghz_state(&qubits).unwrap()
            .build();

        assert_eq!(circuit.operations.len(), 3); // H + 2 CX gates
    }
}