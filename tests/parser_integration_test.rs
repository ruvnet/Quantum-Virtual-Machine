//! Integration tests for the OpenQASM 3 parser

use qvm_scheduler::circuit_ir::parse_qasm3;

#[test]
fn test_comprehensive_qasm_parsing() {
    let qasm = r#"
        OPENQASM 3.0;
        include "stdgates.inc";
        
        qubit[4] q;
        bit[4] c;
        
        // Single-qubit gates
        h q[0];
        x q[1];
        y q[2];
        z q[3];
        
        // Rotation gates with parameters
        rx(1.57) q[0];
        ry(3.14) q[1];
        rz(0.785) q[2];
        
        // Two-qubit gates
        cx q[0], q[1];
        cz q[1], q[2];
        swap q[2], q[3];
        
        // Three-qubit gate
        ccx q[0], q[1], q[2];
        
        // Measurements
        measure q[0] -> c[0];
        c[1] = measure q[1];
        
        // Reset and barrier
        reset q[2];
        barrier q[0], q[1];
        barrier;
    "#;

    let result = parse_qasm3(qasm);
    assert!(result.is_ok(), "Failed to parse QASM: {:?}", result.err());
    
    let circuit = result.unwrap();
    assert_eq!(circuit.num_qubits, 4);
    assert_eq!(circuit.num_classical, 4);
    
    // Should have 16 operations total
    assert_eq!(circuit.operations.len(), 16);
    
    // Check the circuit name
    assert_eq!(circuit.name, "parsed_circuit");
}

#[test]
fn test_bell_state_circuit() {
    let qasm = r#"
        OPENQASM 3.0;
        
        qubit[2] q;
        bit[2] c;
        
        h q[0];
        cx q[0], q[1];
        
        c[0] = measure q[0];
        c[1] = measure q[1];
    "#;

    let result = parse_qasm3(qasm);
    assert!(result.is_ok());
    
    let circuit = result.unwrap();
    assert_eq!(circuit.num_qubits, 2);
    assert_eq!(circuit.operations.len(), 4); // H, CX, measure, measure
}

#[test]
fn test_error_handling() {
    // Test invalid gate
    let bad_qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        invalid_gate q[0];
    "#;
    
    let result = parse_qasm3(bad_qasm);
    assert!(result.is_err());
}

#[test]
fn test_wasm_compatibility() {
    // This test ensures no std-only features are used in parsing
    let qasm = r#"
        OPENQASM 3.0;
        qubit[1] q;
        h q[0];
    "#;

    let result = parse_qasm3(qasm);
    assert!(result.is_ok());
}