//! OpenQASM 3 parser implementation

use crate::{QvmError, Result as QvmResult};
use crate::circuit_ir::{QuantumCircuit, Operation, SingleQubitGate, TwoQubitGate, MultiQubitGate, Qubit, ClassicalBit};
use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, take_until},
    character::complete::{char, digit1, multispace0, space0, space1, alphanumeric1},
    combinator::{map, opt, recognize},
    multi::{many0, separated_list0, separated_list1},
    number::complete::double,
    sequence::{delimited, pair, tuple},
};
use smallvec::SmallVec;
use std::collections::HashMap;

/// Parse a complete OpenQASM 3 program
pub fn parse_qasm3(input: &str) -> QvmResult<QuantumCircuit> {
    let mut parser = QasmParser::new();
    parser.parse(input)
}

/// OpenQASM 3 parser state
#[derive(Debug)]
struct QasmParser {
    /// Qubit register mappings (name -> (size, start_index))
    qubit_registers: HashMap<String, (usize, usize)>,
    /// Classical register mappings (name -> (size, start_index))
    classical_registers: HashMap<String, (usize, usize)>,
    /// Total qubit count
    total_qubits: usize,
    /// Total classical bit count
    total_classical: usize,
    /// Operations collected
    operations: Vec<Operation>,
}

impl QasmParser {
    fn new() -> Self {
        Self {
            qubit_registers: HashMap::new(),
            classical_registers: HashMap::new(),
            total_qubits: 0,
            total_classical: 0,
            operations: Vec::new(),
        }
    }

    /// Parse the complete QASM program
    fn parse(&mut self, input: &str) -> QvmResult<QuantumCircuit> {
        let (remaining, _) = self.parse_program(input)
            .map_err(|e| QvmError::parse_error(format!("Parse error: {:?}", e), 0))?;

        if !remaining.trim().is_empty() {
            return Err(QvmError::parse_error(
                format!("Unexpected content after program: {}", remaining),
                input.len() - remaining.len(),
            ));
        }

        Ok(QuantumCircuit {
            name: "parsed_circuit".to_string(),
            num_qubits: self.total_qubits,
            num_classical: self.total_classical,
            operations: self.operations.clone(),
            metadata: Default::default(),
        })
    }

    /// Parse the complete program structure
    fn parse_program<'a>(&mut self, input: &'a str) -> IResult<&'a str, ()> {
        let (input, _) = multispace0(input)?;
        
        // Optional version declaration
        let (input, _) = opt(|i| self.parse_version(i))(input)?;
        let (input, _) = multispace0(input)?;
        
        // Optional includes
        let (input, _) = many0(|i| self.parse_include(i))(input)?;
        let (input, _) = multispace0(input)?;
        
        // Declarations and statements
        let (input, _) = many0(|i| self.parse_statement(i))(input)?;
        
        Ok((input, ()))
    }

    /// Parse version declaration
    fn parse_version<'a>(&self, input: &'a str) -> IResult<&'a str, ()> {
        let (input, _) = tag("OPENQASM")(input)?;
        let (input, _) = space1(input)?;
        let (input, _) = recognize(pair(digit1, opt(pair(char('.'), digit1))))(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = char(';')(input)?;
        Ok((input, ()))
    }

    /// Parse include statement
    fn parse_include<'a>(&self, input: &'a str) -> IResult<&'a str, ()> {
        let (input, _) = tag("include")(input)?;
        let (input, _) = space1(input)?;
        let (input, _) = delimited(char('"'), take_until("\""), char('"'))(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = char(';')(input)?;
        let (input, _) = multispace0(input)?;
        Ok((input, ()))
    }

    /// Parse a statement (declaration or instruction)
    fn parse_statement<'a>(&mut self, input: &'a str) -> IResult<&'a str, ()> {
        let (input, _) = multispace0(input)?;
        
        let (input, _) = if let Ok((remaining, _)) = self.parse_qubit_declaration(input) {
            (remaining, ())
        } else if let Ok((remaining, _)) = self.parse_classical_declaration(input) {
            (remaining, ())
        } else if let Ok((remaining, _)) = self.parse_gate_instruction(input) {
            (remaining, ())
        } else if let Ok((remaining, _)) = self.parse_measurement(input) {
            (remaining, ())
        } else if let Ok((remaining, _)) = self.parse_reset(input) {
            (remaining, ())
        } else if let Ok((remaining, _)) = self.parse_barrier(input) {
            (remaining, ())
        } else if let Ok((remaining, _)) = self.parse_comment(input) {
            (remaining, ())
        } else {
            return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Alt)));
        };
        
        let (input, _) = multispace0(input)?;
        Ok((input, ()))
    }

    /// Parse qubit declaration
    fn parse_qubit_declaration<'a>(&mut self, input: &'a str) -> IResult<&'a str, ()> {
        let (input, _) = tag("qubit")(input)?;
        let (input, _) = space0(input)?;
        
        // Parse array size [n]
        let (input, size) = opt(delimited(char('['), parse_integer, char(']')))(input)?;
        let size = size.unwrap_or(1);
        
        let (input, _) = space0(input)?;
        let (input, name) = parse_identifier(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = char(';')(input)?;

        // Register the qubits
        self.qubit_registers.insert(name.to_string(), (size, self.total_qubits));
        self.total_qubits += size;

        Ok((input, ()))
    }

    /// Parse classical bit declaration
    fn parse_classical_declaration<'a>(&mut self, input: &'a str) -> IResult<&'a str, ()> {
        let (input, _) = tag("bit")(input)?;
        let (input, _) = space0(input)?;
        
        // Parse array size [n]
        let (input, size) = opt(delimited(char('['), parse_integer, char(']')))(input)?;
        let size = size.unwrap_or(1);
        
        let (input, _) = space0(input)?;
        let (input, name) = parse_identifier(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = char(';')(input)?;

        // Register the classical bits
        self.classical_registers.insert(name.to_string(), (size, self.total_classical));
        self.total_classical += size;

        Ok((input, ()))
    }

    /// Parse gate instruction
    fn parse_gate_instruction<'a>(&mut self, input: &'a str) -> IResult<&'a str, ()> {
        let (input, gate_name) = parse_identifier(input)?;
        let (input, _) = space0(input)?;
        
        // Parse optional parameters
        let (input, parameters) = opt(delimited(
            char('('),
            separated_list0(tuple((space0, char(','), space0)), double),
            char(')')
        ))(input)?;
        let parameters: SmallVec<[f64; 2]> = parameters
            .unwrap_or_default()
            .into_iter()
            .collect();
        
        let (input, _) = space0(input)?;
        
        // Parse qubit arguments
        let (input, qubits) = separated_list1(
            tuple((space0, char(','), space0)),
            |i| self.parse_qubit_ref(i)
        )(input)?;
        
        let (input, _) = space0(input)?;
        let (input, _) = char(';')(input)?;

        // Create operation based on gate name and qubit count
        let operation = match self.create_gate_operation(gate_name, &qubits, parameters) {
            Ok((_, op)) => op,
            Err(_) => return Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag))),
        };
        self.operations.push(operation);

        Ok((input, ()))
    }

    /// Parse measurement instruction
    fn parse_measurement<'a>(&mut self, input: &'a str) -> IResult<&'a str, ()> {
        // Two formats: "measure q[0] -> c[0];" or "c[0] = measure q[0];"
        
        // Try assignment format first
        if let Ok(result) = self.parse_measurement_assignment(input) {
            return Ok(result);
        }
        
        // Try arrow format: "measure q[0] -> c[0];"
        let (input, _) = tag("measure")(input)?;
        let (input, _) = space1(input)?;
        let (input, qubit) = self.parse_qubit_ref(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = tag("->")(input)?;
        let (input, _) = space0(input)?;
        let (input, classical) = self.parse_classical_ref(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = char(';')(input)?;

        self.operations.push(Operation::Measurement { qubit, classical });
        Ok((input, ()))
    }

    /// Parse measurement in assignment format
    fn parse_measurement_assignment<'a>(&mut self, input: &'a str) -> IResult<&'a str, ()> {
        let (input, classical) = self.parse_classical_ref(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = char('=')(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = tag("measure")(input)?;
        let (input, _) = space1(input)?;
        let (input, qubit) = self.parse_qubit_ref(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = char(';')(input)?;

        self.operations.push(Operation::Measurement { qubit, classical });
        Ok((input, ()))
    }

    /// Parse reset instruction
    fn parse_reset<'a>(&mut self, input: &'a str) -> IResult<&'a str, ()> {
        let (input, _) = tag("reset")(input)?;
        let (input, _) = space1(input)?;
        let (input, qubit) = self.parse_qubit_ref(input)?;
        let (input, _) = space0(input)?;
        let (input, _) = char(';')(input)?;

        self.operations.push(Operation::Reset { qubit });
        Ok((input, ()))
    }

    /// Parse barrier instruction
    fn parse_barrier<'a>(&mut self, input: &'a str) -> IResult<&'a str, ()> {
        let (input, _) = tag("barrier")(input)?;
        let (input, _) = space0(input)?;
        
        // Parse optional qubit list
        let (input, qubits) = opt(separated_list1(
            tuple((space0, char(','), space0)),
            |i| self.parse_qubit_ref(i)
        ))(input)?;
        
        let (input, _) = space0(input)?;
        let (input, _) = char(';')(input)?;

        let qubits = qubits.unwrap_or_default().into_iter().collect();
        self.operations.push(Operation::Barrier { qubits });
        Ok((input, ()))
    }

    /// Parse comment
    fn parse_comment<'a>(&self, input: &'a str) -> IResult<&'a str, ()> {
        let (input, _) = tag("//")(input)?;
        let (input, _) = take_until("\n")(input)?;
        Ok((input, ()))
    }

    /// Parse qubit reference (e.g., "q[0]", "qubits[2]")
    fn parse_qubit_ref<'a>(&self, input: &'a str) -> IResult<&'a str, Qubit> {
        let (input, reg_name) = parse_identifier(input)?;
        let (input, _) = space0(input)?;
        let (input, index) = delimited(char('['), parse_integer, char(']'))(input)?;

        // Look up register
        if let Some((_, start_index)) = self.qubit_registers.get(reg_name) {
            Ok((input, Qubit(start_index + index)))
        } else {
            // Return error through IResult
            Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag)))
        }
    }

    /// Parse classical bit reference
    fn parse_classical_ref<'a>(&self, input: &'a str) -> IResult<&'a str, ClassicalBit> {
        let (input, reg_name) = parse_identifier(input)?;
        let (input, _) = space0(input)?;
        let (input, index) = delimited(char('['), parse_integer, char(']'))(input)?;

        // Look up register
        if let Some((_, start_index)) = self.classical_registers.get(reg_name) {
            Ok((input, ClassicalBit(start_index + index)))
        } else {
            Err(nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag)))
        }
    }

    /// Create gate operation from parsed components
    fn create_gate_operation(
        &self,
        gate_name: &str,
        qubits: &[Qubit],
        parameters: SmallVec<[f64; 2]>,
    ) -> IResult<&str, Operation> {
        let operation = match (gate_name, qubits.len()) {
            // Single-qubit gates
            ("i" | "id", 1) => Operation::SingleQubit { gate: SingleQubitGate::I, qubit: qubits[0], parameters },
            ("x", 1) => Operation::SingleQubit { gate: SingleQubitGate::X, qubit: qubits[0], parameters },
            ("y", 1) => Operation::SingleQubit { gate: SingleQubitGate::Y, qubit: qubits[0], parameters },
            ("z", 1) => Operation::SingleQubit { gate: SingleQubitGate::Z, qubit: qubits[0], parameters },
            ("h", 1) => Operation::SingleQubit { gate: SingleQubitGate::H, qubit: qubits[0], parameters },
            ("s", 1) => Operation::SingleQubit { gate: SingleQubitGate::S, qubit: qubits[0], parameters },
            ("sdg", 1) => Operation::SingleQubit { gate: SingleQubitGate::Sdg, qubit: qubits[0], parameters },
            ("t", 1) => Operation::SingleQubit { gate: SingleQubitGate::T, qubit: qubits[0], parameters },
            ("tdg", 1) => Operation::SingleQubit { gate: SingleQubitGate::Tdg, qubit: qubits[0], parameters },
            ("sx", 1) => Operation::SingleQubit { gate: SingleQubitGate::SX, qubit: qubits[0], parameters },
            ("rx", 1) => Operation::SingleQubit { gate: SingleQubitGate::RX, qubit: qubits[0], parameters },
            ("ry", 1) => Operation::SingleQubit { gate: SingleQubitGate::RY, qubit: qubits[0], parameters },
            ("rz", 1) => Operation::SingleQubit { gate: SingleQubitGate::RZ, qubit: qubits[0], parameters },
            ("p", 1) => Operation::SingleQubit { gate: SingleQubitGate::P, qubit: qubits[0], parameters },
            ("u", 1) => Operation::SingleQubit { gate: SingleQubitGate::U, qubit: qubits[0], parameters },
            
            // Two-qubit gates
            ("cx" | "cnot", 2) => Operation::TwoQubit { gate: TwoQubitGate::CNOT, control: qubits[0], target: qubits[1], parameters },
            ("cz", 2) => Operation::TwoQubit { gate: TwoQubitGate::CZ, control: qubits[0], target: qubits[1], parameters },
            ("cy", 2) => Operation::TwoQubit { gate: TwoQubitGate::CY, control: qubits[0], target: qubits[1], parameters },
            ("ch", 2) => Operation::TwoQubit { gate: TwoQubitGate::CH, control: qubits[0], target: qubits[1], parameters },
            ("cp", 2) => Operation::TwoQubit { gate: TwoQubitGate::CP, control: qubits[0], target: qubits[1], parameters },
            ("crx", 2) => Operation::TwoQubit { gate: TwoQubitGate::CRX, control: qubits[0], target: qubits[1], parameters },
            ("cry", 2) => Operation::TwoQubit { gate: TwoQubitGate::CRY, control: qubits[0], target: qubits[1], parameters },
            ("crz", 2) => Operation::TwoQubit { gate: TwoQubitGate::CRZ, control: qubits[0], target: qubits[1], parameters },
            ("swap", 2) => Operation::TwoQubit { gate: TwoQubitGate::SWAP, control: qubits[0], target: qubits[1], parameters },
            ("iswap", 2) => Operation::TwoQubit { gate: TwoQubitGate::ISWAP, control: qubits[0], target: qubits[1], parameters },
            
            // Three-qubit gates
            ("ccx" | "toffoli", 3) => {
                let mut multi_qubits = SmallVec::new();
                multi_qubits.extend_from_slice(qubits);
                let mut multi_params = SmallVec::new();
                multi_params.extend_from_slice(&parameters);
                Operation::MultiQubit { gate: MultiQubitGate::Toffoli, qubits: multi_qubits, parameters: multi_params }
            }
            
            _ => return Err(nom::Err::Error(nom::error::Error::new("", nom::error::ErrorKind::Tag))),
        };
        
        Ok(("", operation))
    }
}

/// Parse identifier (alphanumeric + underscore, starting with letter or underscore)
fn parse_identifier(input: &str) -> IResult<&str, &str> {
    recognize(pair(
        alt((alphanumeric1, tag("_"))),
        many0(alt((alphanumeric1, tag("_"))))
    ))(input)
}

/// Parse integer
fn parse_integer(input: &str) -> IResult<&str, usize> {
    map(digit1, |s: &str| s.parse().unwrap_or(0))(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_circuit_parsing() {
        let qasm = r#"
            OPENQASM 3.0;
            include "stdgates.inc";
            
            qubit[2] q;
            bit[2] c;
            
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.num_classical, 2);
        assert_eq!(circuit.operations.len(), 4);
    }

    #[test]
    fn test_gate_parsing() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            
            x q[0];
            h q[0];
            rx(1.57) q[0];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.operations.len(), 3);
        
        // Check first gate is X
        match &circuit.operations[0] {
            Operation::SingleQubit { gate: SingleQubitGate::X, qubit, .. } => {
                assert_eq!(*qubit, Qubit(0));
            }
            _ => panic!("Expected X gate"),
        }
    }

    #[test]
    fn test_two_qubit_gate() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            
            cx q[0], q[1];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.operations.len(), 1);
        
        match &circuit.operations[0] {
            Operation::TwoQubit { gate: TwoQubitGate::CNOT, control, target, .. } => {
                assert_eq!(*control, Qubit(0));
                assert_eq!(*target, Qubit(1));
            }
            _ => panic!("Expected CNOT gate"),
        }
    }

    #[test]
    fn test_comprehensive_single_qubit_gates() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            
            i q[0];
            x q[0];
            y q[0];
            z q[0];
            h q[0];
            s q[0];
            sdg q[0];
            t q[0];
            tdg q[0];
            sx q[0];
            rx(1.57) q[0];
            ry(3.14) q[0];
            rz(0.78) q[0];
            p(1.0) q[0];
            u(1.0, 2.0, 3.0) q[0];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.operations.len(), 15);
        assert_eq!(circuit.num_qubits, 1);
    }

    #[test]
    fn test_comprehensive_two_qubit_gates() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            
            cx q[0], q[1];
            cnot q[0], q[1];
            cz q[0], q[1];
            cy q[0], q[1];
            ch q[0], q[1];
            cp(1.57) q[0], q[1];
            crx(1.0) q[0], q[1];
            cry(2.0) q[0], q[1];
            crz(3.0) q[0], q[1];
            swap q[0], q[1];
            iswap q[0], q[1];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.operations.len(), 11);
        assert_eq!(circuit.num_qubits, 2);
    }

    #[test]
    fn test_three_qubit_gates() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            
            ccx q[0], q[1], q[2];
            toffoli q[0], q[1], q[2];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.operations.len(), 2);
        
        // Check both are Toffoli gates
        for op in &circuit.operations {
            match op {
                Operation::MultiQubit { gate: MultiQubitGate::Toffoli, qubits, .. } => {
                    assert_eq!(qubits.len(), 3);
                }
                _ => panic!("Expected Toffoli gate"),
            }
        }
    }

    #[test]
    fn test_measurement_formats() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            bit[2] c;
            
            measure q[0] -> c[0];
            c[1] = measure q[1];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.operations.len(), 2);
        
        // Check both are measurement operations
        for op in &circuit.operations {
            match op {
                Operation::Measurement { .. } => {} // Expected
                _ => panic!("Expected measurement operation"),
            }
        }
    }

    #[test]
    fn test_reset_and_barrier() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[3] q;
            
            reset q[0];
            barrier q[0], q[1];
            barrier;
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.operations.len(), 3);
        
        // Check reset
        match &circuit.operations[0] {
            Operation::Reset { qubit } => assert_eq!(*qubit, Qubit(0)),
            _ => panic!("Expected reset operation"),
        }
        
        // Check barrier with qubits
        match &circuit.operations[1] {
            Operation::Barrier { qubits } => assert_eq!(qubits.len(), 2),
            _ => panic!("Expected barrier operation"),
        }
        
        // Check barrier without qubits
        match &circuit.operations[2] {
            Operation::Barrier { qubits } => assert_eq!(qubits.len(), 0),
            _ => panic!("Expected barrier operation"),
        }
    }

    #[test]
    fn test_comments() {
        let qasm = r#"
            OPENQASM 3.0;
            // This is a comment
            qubit[1] q;
            
            x q[0];  // Another comment
            // Final comment
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.operations.len(), 1);
    }

    #[test]
    fn test_parameter_parsing() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[2] q;
            
            rx(3.14159) q[0];
            cry(1.5708, 2.7183) q[0], q[1];
        "#;

        let circuit = parse_qasm3(qasm).unwrap();
        assert_eq!(circuit.operations.len(), 2);
        
        // Check RX parameter
        match &circuit.operations[0] {
            Operation::SingleQubit { parameters, .. } => {
                assert!((parameters[0] - 3.14159).abs() < 1e-5);
            }
            _ => panic!("Expected single qubit operation"),
        }
    }

    #[test]
    fn test_malformed_circuit() {
        let bad_qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            
            invalid_gate q[0];
        "#;

        let result = parse_qasm3(bad_qasm);
        assert!(result.is_err());
    }

    #[test]
    fn test_qubit_out_of_range() {
        let qasm = r#"
            OPENQASM 3.0;
            qubit[1] q;
            
            x q[5];
        "#;

        let result = parse_qasm3(qasm);
        assert!(result.is_err());
    }
}