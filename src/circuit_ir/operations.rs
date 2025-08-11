//! Quantum operations and gate definitions

use crate::{Qubit, ClassicalBit};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

/// Quantum operation variants
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    /// Single-qubit gate
    SingleQubit {
        gate: SingleQubitGate,
        qubit: Qubit,
        parameters: SmallVec<[f64; 2]>,
    },
    /// Two-qubit gate
    TwoQubit {
        gate: TwoQubitGate,
        control: Qubit,
        target: Qubit,
        parameters: SmallVec<[f64; 2]>,
    },
    /// Multi-qubit gate (for future expansion)
    MultiQubit {
        gate: MultiQubitGate,
        qubits: SmallVec<[Qubit; 4]>,
        parameters: SmallVec<[f64; 4]>,
    },
    /// Measurement operation
    Measurement {
        qubit: Qubit,
        classical: ClassicalBit,
    },
    /// Reset operation
    Reset {
        qubit: Qubit,
    },
    /// Barrier for synchronization
    Barrier {
        qubits: SmallVec<[Qubit; 4]>,
    },
    /// Classical operation (for future expansion)
    Classical {
        operation: ClassicalOperation,
    },
}

/// Single-qubit gates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SingleQubitGate {
    /// Identity gate
    I,
    /// Pauli-X (NOT) gate
    X,
    /// Pauli-Y gate
    Y,
    /// Pauli-Z gate
    Z,
    /// Hadamard gate
    H,
    /// S gate (phase)
    S,
    /// S-dagger gate
    Sdg,
    /// T gate (π/8 phase)
    T,
    /// T-dagger gate
    Tdg,
    /// Rotation around X-axis
    RX,
    /// Rotation around Y-axis
    RY,
    /// Rotation around Z-axis
    RZ,
    /// Phase gate
    P,
    /// Square root of X
    SX,
    /// U gate (general single-qubit unitary)
    U,
}

/// Two-qubit gates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TwoQubitGate {
    /// Controlled-NOT gate
    CNOT,
    /// Controlled-Z gate
    CZ,
    /// Controlled-Y gate
    CY,
    /// Controlled-H gate
    CH,
    /// Controlled phase gate
    CP,
    /// Controlled RX gate
    CRX,
    /// Controlled RY gate
    CRY,
    /// Controlled RZ gate
    CRZ,
    /// Controlled U gate
    CU,
    /// SWAP gate
    SWAP,
    /// iSWAP gate
    ISWAP,
    /// Square root of SWAP
    SQRTSWAP,
    /// Molmer-Sorensen gate
    MS,
    /// XX interaction
    XX,
    /// YY interaction
    YY,
    /// ZZ interaction
    ZZ,
}

/// Multi-qubit gates (for future expansion)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MultiQubitGate {
    /// Toffoli (CCX) gate
    Toffoli,
    /// Fredkin (CSWAP) gate
    Fredkin,
    /// Multi-controlled X
    MCX,
    /// Multi-controlled Z
    MCZ,
    /// Quantum Fourier Transform
    QFT,
}

/// Classical operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ClassicalOperation {
    /// Assign constant value
    Assign {
        target: ClassicalBit,
        value: bool,
    },
    /// Copy from another bit
    Copy {
        source: ClassicalBit,
        target: ClassicalBit,
    },
    /// Logical AND
    And {
        left: ClassicalBit,
        right: ClassicalBit,
        target: ClassicalBit,
    },
    /// Logical OR
    Or {
        left: ClassicalBit,
        right: ClassicalBit,
        target: ClassicalBit,
    },
    /// Logical NOT
    Not {
        source: ClassicalBit,
        target: ClassicalBit,
    },
}

impl Operation {
    /// Get all qubits involved in this operation
    pub fn qubits(&self) -> SmallVec<[Qubit; 4]> {
        match self {
            Self::SingleQubit { qubit, .. } => {
                let mut result = SmallVec::new();
                result.push(*qubit);
                result
            }
            Self::TwoQubit { control, target, .. } => {
                let mut result = SmallVec::new();
                result.push(*control);
                result.push(*target);
                result
            }
            Self::MultiQubit { qubits, .. } => qubits.clone(),
            Self::Measurement { qubit, .. } => {
                let mut result = SmallVec::new();
                result.push(*qubit);
                result
            }
            Self::Reset { qubit } => {
                let mut result = SmallVec::new();
                result.push(*qubit);
                result
            }
            Self::Barrier { qubits } => qubits.clone(),
            Self::Classical { .. } => SmallVec::new(),
        }
    }

    /// Get classical bits involved in this operation
    pub fn classical_bits(&self) -> SmallVec<[ClassicalBit; 4]> {
        match self {
            Self::Measurement { classical, .. } => {
                let mut result = SmallVec::new();
                result.push(*classical);
                result
            }
            Self::Classical { operation } => {
                match operation {
                    ClassicalOperation::Assign { target, .. } => {
                        let mut result = SmallVec::new();
                        result.push(*target);
                        result
                    }
                    ClassicalOperation::Copy { source, target } => {
                        let mut result = SmallVec::new();
                        result.push(*source);
                        result.push(*target);
                        result
                    }
                    ClassicalOperation::And { left, right, target } |
                    ClassicalOperation::Or { left, right, target } => {
                        let mut result = SmallVec::new();
                        result.push(*left);
                        result.push(*right);
                        result.push(*target);
                        result
                    }
                    ClassicalOperation::Not { source, target } => {
                        let mut result = SmallVec::new();
                        result.push(*source);
                        result.push(*target);
                        result
                    }
                }
            }
            _ => SmallVec::new(),
        }
    }

    /// Check if this operation is a two-qubit gate
    pub fn is_two_qubit(&self) -> bool {
        matches!(self, Self::TwoQubit { .. })
    }

    /// Check if this operation involves a specific qubit
    pub fn involves_qubit(&self, qubit: Qubit) -> bool {
        self.qubits().contains(&qubit)
    }

    /// Convert operation to OpenQASM 3 representation
    pub fn to_qasm(&self) -> String {
        match self {
            Self::SingleQubit { gate, qubit, parameters } => {
                match gate {
                    SingleQubitGate::I => format!("id q[{}];", qubit.index()),
                    SingleQubitGate::X => format!("x q[{}];", qubit.index()),
                    SingleQubitGate::Y => format!("y q[{}];", qubit.index()),
                    SingleQubitGate::Z => format!("z q[{}];", qubit.index()),
                    SingleQubitGate::H => format!("h q[{}];", qubit.index()),
                    SingleQubitGate::S => format!("s q[{}];", qubit.index()),
                    SingleQubitGate::Sdg => format!("sdg q[{}];", qubit.index()),
                    SingleQubitGate::T => format!("t q[{}];", qubit.index()),
                    SingleQubitGate::Tdg => format!("tdg q[{}];", qubit.index()),
                    SingleQubitGate::RX => {
                        let angle = parameters.get(0).unwrap_or(&0.0);
                        format!("rx({}) q[{}];", angle, qubit.index())
                    }
                    SingleQubitGate::RY => {
                        let angle = parameters.get(0).unwrap_or(&0.0);
                        format!("ry({}) q[{}];", angle, qubit.index())
                    }
                    SingleQubitGate::RZ => {
                        let angle = parameters.get(0).unwrap_or(&0.0);
                        format!("rz({}) q[{}];", angle, qubit.index())
                    }
                    SingleQubitGate::P => {
                        let angle = parameters.get(0).unwrap_or(&0.0);
                        format!("p({}) q[{}];", angle, qubit.index())
                    }
                    SingleQubitGate::SX => format!("sx q[{}];", qubit.index()),
                    SingleQubitGate::U => {
                        let theta = parameters.get(0).unwrap_or(&0.0);
                        let phi = parameters.get(1).unwrap_or(&0.0);
                        let lambda = parameters.get(2).unwrap_or(&0.0);
                        format!("u({}, {}, {}) q[{}];", theta, phi, lambda, qubit.index())
                    }
                }
            }
            Self::TwoQubit { gate, control, target, parameters } => {
                match gate {
                    TwoQubitGate::CNOT => format!("cx q[{}], q[{}];", control.index(), target.index()),
                    TwoQubitGate::CZ => format!("cz q[{}], q[{}];", control.index(), target.index()),
                    TwoQubitGate::CY => format!("cy q[{}], q[{}];", control.index(), target.index()),
                    TwoQubitGate::CH => format!("ch q[{}], q[{}];", control.index(), target.index()),
                    TwoQubitGate::CP => {
                        let angle = parameters.get(0).unwrap_or(&0.0);
                        format!("cp({}) q[{}], q[{}];", angle, control.index(), target.index())
                    }
                    TwoQubitGate::SWAP => format!("swap q[{}], q[{}];", control.index(), target.index()),
                    TwoQubitGate::ISWAP => format!("iswap q[{}], q[{}];", control.index(), target.index()),
                    _ => format!("// Unsupported gate: {:?}", gate),
                }
            }
            Self::Measurement { qubit, classical } => {
                format!("c[{}] = measure q[{}];", classical.index(), qubit.index())
            }
            Self::Reset { qubit } => {
                format!("reset q[{}];", qubit.index())
            }
            Self::Barrier { qubits } => {
                if qubits.is_empty() {
                    "barrier;".to_string()
                } else {
                    let qubit_list: Vec<String> = qubits
                        .iter()
                        .map(|q| format!("q[{}]", q.index()))
                        .collect();
                    format!("barrier {};", qubit_list.join(", "))
                }
            }
            Self::MultiQubit { gate, qubits, .. } => {
                match gate {
                    MultiQubitGate::Toffoli => {
                        if qubits.len() >= 3 {
                            format!("ccx q[{}], q[{}], q[{}];", 
                                   qubits[0].index(), qubits[1].index(), qubits[2].index())
                        } else {
                            "// Invalid Toffoli gate".to_string()
                        }
                    }
                    _ => format!("// Unsupported multi-qubit gate: {:?}", gate),
                }
            }
            Self::Classical { .. } => {
                "// Classical operation (not yet implemented)".to_string()
            }
        }
    }

    /// Get estimated execution time for this operation
    pub fn estimated_time(&self) -> f64 {
        match self {
            Self::SingleQubit { .. } => 10e-9, // 10 ns
            Self::TwoQubit { .. } => 50e-9,    // 50 ns
            Self::MultiQubit { .. } => 100e-9, // 100 ns
            Self::Measurement { .. } => 1e-6,  // 1 μs
            Self::Reset { .. } => 1e-6,        // 1 μs
            Self::Barrier { .. } => 0.0,       // Instantaneous
            Self::Classical { .. } => 1e-9,    // 1 ns
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_qubit_operation() {
        let op = Operation::SingleQubit {
            gate: SingleQubitGate::X,
            qubit: Qubit(0),
            parameters: SmallVec::new(),
        };

        let expected: SmallVec<[Qubit; 4]> = SmallVec::from_slice(&[Qubit(0)]);
        assert_eq!(op.qubits(), expected);
        assert!(op.involves_qubit(Qubit(0)));
        assert!(!op.involves_qubit(Qubit(1)));
        assert!(!op.is_two_qubit());
    }

    #[test]
    fn test_two_qubit_operation() {
        let op = Operation::TwoQubit {
            gate: TwoQubitGate::CNOT,
            control: Qubit(0),
            target: Qubit(1),
            parameters: SmallVec::new(),
        };

        let expected: SmallVec<[Qubit; 4]> = SmallVec::from_slice(&[Qubit(0), Qubit(1)]);
        assert_eq!(op.qubits(), expected);
        assert!(op.involves_qubit(Qubit(0)));
        assert!(op.involves_qubit(Qubit(1)));
        assert!(!op.involves_qubit(Qubit(2)));
        assert!(op.is_two_qubit());
    }

    #[test]
    fn test_qasm_generation() {
        let op = Operation::SingleQubit {
            gate: SingleQubitGate::H,
            qubit: Qubit(0),
            parameters: SmallVec::new(),
        };

        assert_eq!(op.to_qasm(), "h q[0];");
    }

    #[test]
    fn test_measurement_qasm() {
        let op = Operation::Measurement {
            qubit: Qubit(0),
            classical: ClassicalBit(0),
        };

        assert_eq!(op.to_qasm(), "c[0] = measure q[0];");
    }
}