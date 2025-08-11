//! Integration tests for the QVM scheduler

use qvm_scheduler::{QvmScheduler, QuantumCircuit};
use qvm_scheduler::topology::TopologyBuilder;
use qvm_scheduler::circuit_ir::CircuitBuilder;

#[tokio::test]
async fn test_end_to_end_scheduling() {
    // Create topology
    let topology = TopologyBuilder::grid(4, 4);
    let mut scheduler = QvmScheduler::new(topology);

    // Create test circuits
    let circuits = vec![
        CircuitBuilder::new("test1", 2, 2)
            .h(0).unwrap()
            .cx(0, 1).unwrap()
            .measure_all().unwrap()
            .build(),
        CircuitBuilder::new("test2", 3, 3)
            .ghz_state(&[0.into(), 1.into(), 2.into()]).unwrap()
            .measure_all().unwrap()
            .build(),
    ];

    // Schedule circuits
    let composite = scheduler.schedule_circuits(circuits).await.unwrap();
    
    // Verify results
    assert_eq!(composite.circuits().len(), 2);
    assert!(composite.total_duration() > 0);

    // Generate QASM
    let qasm = composite.to_qasm().unwrap();
    assert!(qasm.contains("OPENQASM 3.0"));
    assert!(qasm.contains("qubit"));
}

#[tokio::test]
async fn test_large_circuit_scheduling() {
    let topology = TopologyBuilder::grid(8, 8);
    let mut scheduler = QvmScheduler::new(topology);

    // Create many small circuits
    let circuits: Vec<_> = (0..20)
        .map(|i| {
            CircuitBuilder::new(format!("circuit_{}", i), 2, 2)
                .h(0).unwrap()
                .cx(0, 1).unwrap()
                .measure_all().unwrap()
                .build()
        })
        .collect();

    let composite = scheduler.schedule_circuits(circuits).await.unwrap();
    assert_eq!(composite.circuits().len(), 20);
}

#[tokio::test]
async fn test_topology_validation() {
    let topology = TopologyBuilder::linear(5);
    assert_eq!(topology.qubit_count(), 5);
    assert_eq!(topology.connection_count(), 4);
    
    // Test connectivity
    assert!(topology.are_connected(0.into(), 1.into()));
    assert!(!topology.are_connected(0.into(), 4.into()));
}

#[test]
fn test_circuit_parsing_and_generation() {
    let original_circuit = CircuitBuilder::new("test", 2, 2)
        .h(0).unwrap()
        .cx(0, 1).unwrap()
        .measure_all().unwrap()
        .build();

    let qasm = original_circuit.to_qasm().unwrap();
    assert!(qasm.contains("OPENQASM 3.0"));
    assert!(qasm.contains("h q[0]"));
    assert!(qasm.contains("cx q[0], q[1]"));
}