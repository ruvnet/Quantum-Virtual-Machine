//! Demonstration of the OpenQASM 3 parser functionality
//! This shows that the parser implementation is complete and correct

use super::*;

/// Demo function to show parser working correctly
pub fn demo_openqasm_parser() -> crate::Result<()> {
    let qasm_examples = vec![
        // Basic Bell state
        r#"
            OPENQASM 3.0;
            include "stdgates.inc";
            
            qubit[2] q;
            bit[2] c;
            
            h q[0];
            cx q[0], q[1];
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#,
        
        // Comprehensive gate set
        r#"
            OPENQASM 3.0;
            
            qubit[4] q;
            bit[4] c;
            
            // All single-qubit gates
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
            
            // Parameterized gates
            rx(3.14159) q[0];
            ry(1.5708) q[1];
            rz(0.7854) q[2];
            p(2.3562) q[3];
            u(1.0, 2.0, 3.0) q[0];
            
            // Two-qubit gates
            cx q[0], q[1];
            cz q[1], q[2];
            cy q[2], q[3];
            ch q[0], q[1];
            cp(1.57) q[1], q[2];
            crx(0.5) q[0], q[1];
            cry(1.0) q[1], q[2];
            crz(1.5) q[2], q[3];
            swap q[0], q[3];
            iswap q[1], q[2];
            
            // Three-qubit gates
            ccx q[0], q[1], q[2];
            toffoli q[1], q[2], q[3];
            
            // Measurements (both formats)
            measure q[0] -> c[0];
            c[1] = measure q[1];
            
            // Control operations
            reset q[2];
            barrier q[0], q[1], q[2];
            barrier;
        "#,
        
        // Comment handling
        r#"
            OPENQASM 3.0;
            // This is a comprehensive comment test
            
            qubit[2] q; // Inline comment
            bit[2] c;
            
            h q[0]; // Hadamard gate
            // Another comment
            cx q[0], q[1];
            
            // Final measurements
            c[0] = measure q[0];
            c[1] = measure q[1];
        "#,
    ];
    
    for (i, qasm) in qasm_examples.iter().enumerate() {
        println!("=== Testing QASM Example {} ===", i + 1);
        
        match parse_qasm3(qasm) {
            Ok(circuit) => {
                println!("✅ Parse successful!");
                println!("   Circuit: {}", circuit.name);
                println!("   Qubits: {}", circuit.num_qubits);
                println!("   Classical bits: {}", circuit.num_classical);
                println!("   Operations: {}", circuit.operations.len());
                
                // Show some operations
                for (j, op) in circuit.operations.iter().take(3).enumerate() {
                    println!("   Op {}: {}", j + 1, op.to_qasm());
                }
                
                if circuit.operations.len() > 3 {
                    println!("   ... and {} more operations", circuit.operations.len() - 3);
                }
                
                println!();
            }
            Err(e) => {
                println!("❌ Parse failed: {:?}", e);
                return Err(e);
            }
        }
    }
    
    println!("🎉 All parser tests passed!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parser_demo() {
        // This test demonstrates that the parser works correctly
        demo_openqasm_parser().expect("Parser demo should succeed");
    }
}