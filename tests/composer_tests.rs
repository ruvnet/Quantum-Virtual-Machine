//! Comprehensive tests for the circuit composer system

use quantum_virtual_machine::*;
use quantum_virtual_machine::circuit_ir::*;
use quantum_virtual_machine::topology::TopologyBuilder;
use quantum_virtual_machine::scheduler::*;
use quantum_virtual_machine::composer::*;
use quantum_virtual_machine::composer::mapping::*;
use quantum_virtual_machine::composer::timing::*;
use quantum_virtual_machine::composer::output::*;
use smallvec::SmallVec;

#[tokio::test]
async fn test_complete_circuit_composition_workflow() {
    let topology = TopologyBuilder::grid(3, 3);
    let mut composer = CircuitComposer::new(topology.clone());
    
    // Create test circuits
    let mut circuit1 = QuantumCircuit::new("test1".to_string(), 2, 1);
    circuit1.add_operation(Operation::SingleQubit {
        gate: SingleQubitGate::H,
        qubit: Qubit(0),
        parameters: SmallVec::new(),
    });
    circuit1.add_operation(Operation::TwoQubit {
        gate: TwoQubitGate::CNOT,
        control: Qubit(0),
        target: Qubit(1),
        parameters: SmallVec::new(),
    });
    circuit1.add_operation(Operation::Measurement {
        qubit: Qubit(1),
        classical: ClassicalBit(0),
    });

    let mut circuit2 = QuantumCircuit::new("test2".to_string(), 2, 1);
    circuit2.add_operation(Operation::SingleQubit {
        gate: SingleQubitGate::X,
        qubit: Qubit(0),
        parameters: SmallVec::new(),
    });
    
    // Create schedule with assignments
    let assignment1 = Assignment::new(0, 1, 0, 2000)
        .with_qubit_mapping(vec![0, 1])
        .with_classical_mapping(vec![0]);
    
    let assignment2 = Assignment::new(1, 1, 3000, 1000)
        .with_qubit_mapping(vec![2, 3])
        .with_classical_mapping(vec![1]);
    
    let schedule = Schedule {
        assignments: vec![assignment1, assignment2],
        metadata: ScheduleMetadata::default(),
        total_duration: 4000,
        utilization: ResourceUtilization::default(),
    };

    // Compose the schedule
    let composite = composer.compose(schedule).await.unwrap();
    
    assert_eq!(composite.circuit_count(), 2);
    assert_eq!(composite.total_duration(), 4000);
    assert!(composite.resource_summary.total_qubits_used >= 2);
}

#[test]
fn test_qubit_mapper_strategies() {
    let topology = TopologyBuilder::grid(3, 3);
    let mapper = QubitMapper::new(&topology);
    
    // Test linear mapping
    let linear_mapping = mapper.linear_mapping(4).unwrap();
    assert_eq!(linear_mapping, vec![0, 1, 2, 3]);
    
    // Test connectivity-aware mapping
    let connectivity_requirements = vec![(0, 1), (1, 2), (2, 3)];
    let aware_mapping = mapper.connectivity_aware_mapping(4, &connectivity_requirements).unwrap();
    assert_eq!(aware_mapping.len(), 4);
    
    // Test distance-minimizing mapping
    let distance_mapping = mapper.distance_minimizing_mapping(3, &connectivity_requirements).unwrap();
    assert_eq!(distance_mapping.len(), 3);
}

#[test]
fn test_classical_allocator() {
    let mut allocator = ClassicalAllocator::new(8);
    
    // Test sequential allocation
    let bits1 = allocator.allocate_bits(3).unwrap();
    assert_eq!(bits1, vec![0, 1, 2]);
    assert_eq!(allocator.available_bits(), 5);
    
    let bits2 = allocator.allocate_bits(2).unwrap();
    assert_eq!(bits2, vec![3, 4]);
    assert_eq!(allocator.available_bits(), 3);
    
    // Test release
    allocator.release_bits(&bits1);
    assert_eq!(allocator.available_bits(), 6);
    
    // Test allocation after release
    let bits3 = allocator.allocate_bits(2).unwrap();
    assert!(bits3.iter().all(|&bit| bit <= 2 || bit >= 5));
}

#[test]
fn test_circuit_timing_operations() {
    let timer = CircuitTimer::new();
    
    // Test timing creation
    let timing = timer.create_timing(1000, 2000);
    assert_eq!(timing.start_time, 1000);
    assert_eq!(timing.duration, 2000);
    assert_eq!(timing.estimated_end_time, 3000);
    
    // Test overlap detection
    let timing1 = CircuitTiming::new(1000, 2000);
    let timing2 = CircuitTiming::new(2500, 1000);
    let timing3 = CircuitTiming::new(4000, 1000);
    
    assert!(timing1.overlaps_with(&timing2));
    assert!(!timing1.overlaps_with(&timing3));
    
    // Test gap calculation
    assert_eq!(timing1.gap_to(&timing3), 1000);
    assert_eq!(timing3.gap_to(&timing1), -1000);
}

#[test]
fn test_timing_optimization() {
    let timer = CircuitTimer::new();
    
    let mut timings = vec![
        CircuitTiming::new(1000, 1000),
        CircuitTiming::new(5000, 1500), // Large gap
        CircuitTiming::new(8000, 500),
    ];

    timer.optimize_timings(&mut timings).unwrap();
    
    // Should compress timeline
    assert_eq!(timings[0].start_time, 0);
    assert!(timings[1].start_time >= timings[0].estimated_end_time);
    assert!(timings[2].start_time >= timings[1].estimated_end_time);
}

#[test]
fn test_timing_validation() {
    let timer = CircuitTimer::new();
    
    let timings_with_overlap = vec![
        CircuitTiming::new(1000, 2000),
        CircuitTiming::new(1500, 1000), // Overlaps with first
    ];

    let violations = timer.validate_timings(&timings_with_overlap).unwrap();
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].violation_type, ViolationType::Overlap);
    assert_eq!(violations[0].severity, Severity::Critical);
}

#[test]
fn test_output_generator_qasm3() {
    let generator = OutputGenerator::new();
    let mut composite = CompositeCircuit::new();

    // Create a simple circuit
    let mut circuit = QuantumCircuit::new("test".to_string(), 2, 1);
    circuit.add_operation(Operation::SingleQubit {
        gate: SingleQubitGate::H,
        qubit: Qubit(0),
        parameters: SmallVec::new(),
    });
    circuit.add_operation(Operation::TwoQubit {
        gate: TwoQubitGate::CNOT,
        control: Qubit(0),
        target: Qubit(1),
        parameters: SmallVec::new(),
    });
    circuit.add_operation(Operation::Measurement {
        qubit: Qubit(1),
        classical: ClassicalBit(0),
    });

    let composed_circuit = ComposedCircuit {
        job_id: 0,
        circuit,
        timing: CircuitTiming::new(0, 1000),
        mapping: ResourceMapping {
            qubit_mapping: vec![0, 1],
            classical_mapping: vec![0],
            metadata: Default::default(),
        },
    };

    composite.add_circuit(composed_circuit);
    
    let qasm = generator.generate_qasm(&composite).unwrap();
    
    assert!(qasm.contains("OPENQASM 3.0"));
    assert!(qasm.contains("include \"stdgates.inc\""));
    assert!(qasm.contains("qubit[2] q"));
    assert!(qasm.contains("bit[1] c"));
    assert!(qasm.contains("h q[0]"));
    assert!(qasm.contains("cx q[0], q[1]"));
    assert!(qasm.contains("measure q[1] -> c[0]"));
}

#[test]
fn test_output_generator_with_mapping() {
    let generator = OutputGenerator::new();
    let mapping = vec![2, 5, 1]; // logical 0->physical 2, logical 1->physical 5, etc.
    
    let original = "h q[0]; cx q[0], q[1]; measure q[1] -> c[0];";
    let mapped = generator.apply_qubit_mapping(original, &mapping).unwrap();
    
    assert!(mapped.contains("h q[2]"));
    assert!(mapped.contains("cx q[2], q[5]"));
    assert!(mapped.contains("measure q[5]"));
}

#[test]
fn test_composite_circuit_validation() {
    let topology = TopologyBuilder::grid(2, 2);
    let composer = CircuitComposer::new(topology);
    let mut composite = CompositeCircuit::new();

    // Add two circuits with overlapping timing and resources
    let circuit1 = ComposedCircuit {
        job_id: 0,
        circuit: QuantumCircuit::new("test1".to_string(), 2, 1),
        timing: CircuitTiming::new(1000, 2000), // 1000-3000
        mapping: ResourceMapping {
            qubit_mapping: vec![0, 1],
            classical_mapping: vec![0],
            metadata: Default::default(),
        },
    };

    let circuit2 = ComposedCircuit {
        job_id: 1,
        circuit: QuantumCircuit::new("test2".to_string(), 2, 1),
        timing: CircuitTiming::new(2000, 1500), // 2000-3500 (overlaps!)
        mapping: ResourceMapping {
            qubit_mapping: vec![0, 1], // Same qubits!
            classical_mapping: vec![0],
            metadata: Default::default(),
        },
    };

    composite.add_circuit(circuit1);
    composite.add_circuit(circuit2);

    let validation_report = composer.validate_composite(&composite).unwrap();
    
    assert!(!validation_report.is_valid);
    assert!(!validation_report.violations.is_empty());
    assert!(validation_report.violations[0].contains("overlapping timing"));
}

#[test]
fn test_buffer_and_reset_insertion() {
    let topology = TopologyBuilder::grid(2, 2);
    let mut composer = CircuitComposer::new(topology);
    let mut composite = CompositeCircuit::new();

    // Create circuits that share qubits
    let circuit1 = ComposedCircuit {
        job_id: 0,
        circuit: QuantumCircuit::new("test1".to_string(), 2, 1),
        timing: CircuitTiming::new(0, 1000),
        mapping: ResourceMapping {
            qubit_mapping: vec![0, 1],
            classical_mapping: vec![0],
            metadata: Default::default(),
        },
    };

    let mut circuit2 = ComposedCircuit {
        job_id: 1,
        circuit: QuantumCircuit::new("test2".to_string(), 2, 1),
        timing: CircuitTiming::new(2000, 1000),
        mapping: ResourceMapping {
            qubit_mapping: vec![0, 2], // Shares qubit 0
            classical_mapping: vec![1],
            metadata: Default::default(),
        },
    };

    composite.add_circuit(circuit1);
    composite.add_circuit(circuit2);

    // Insert buffers and reset operations
    composer.insert_buffers(&mut composite).unwrap();
    composer.insert_reset_operations(&mut composite).unwrap();

    // Check that reset operation was inserted
    let second_circuit = &composite.circuits()[1];
    assert!(!second_circuit.circuit.operations.is_empty());
    
    // First operation should be either Reset or Barrier
    match &second_circuit.circuit.operations[0] {
        Operation::Reset { .. } | Operation::Barrier { .. } => {},
        _ => panic!("Expected Reset or Barrier operation to be inserted"),
    }
}

#[test]
fn test_qasm_output_with_timing() {
    let config = OutputConfig {
        format: OutputFormat::OpenQASM3,
        include_timing: true,
        include_mapping: true,
        include_metadata: true,
        optimization_level: 2,
    };
    
    let generator = OutputGenerator::with_config(config);
    let mut composite = CompositeCircuit::new();

    let composed_circuit = ComposedCircuit {
        job_id: 0,
        circuit: QuantumCircuit::new("test".to_string(), 1, 1),
        timing: CircuitTiming::new(1000, 500), // Non-zero start time
        mapping: ResourceMapping {
            qubit_mapping: vec![0],
            classical_mapping: vec![0],
            metadata: Default::default(),
        },
    };

    composite.add_circuit(composed_circuit);
    
    let qasm = generator.generate_qasm(&composite).unwrap();
    
    assert!(qasm.contains("// Start at: 1000 us"));
    assert!(qasm.contains("delay[1000us] q"));
    assert!(qasm.contains("barrier q"));
}

#[test]
fn test_resource_summary_calculation() {
    let mut composite = CompositeCircuit::new();

    let circuit1 = ComposedCircuit {
        job_id: 0,
        circuit: QuantumCircuit::new("test1".to_string(), 2, 1),
        timing: CircuitTiming::new(0, 1000),
        mapping: ResourceMapping {
            qubit_mapping: vec![0, 2], // Uses qubits 0, 2
            classical_mapping: vec![0],
            metadata: Default::default(),
        },
    };

    let circuit2 = ComposedCircuit {
        job_id: 1,
        circuit: QuantumCircuit::new("test2".to_string(), 2, 1),
        timing: CircuitTiming::new(1000, 1000),
        mapping: ResourceMapping {
            qubit_mapping: vec![1, 3], // Uses qubits 1, 3
            classical_mapping: vec![1],
            metadata: Default::default(),
        },
    };

    composite.add_circuit(circuit1);
    composite.add_circuit(circuit2);

    // Resource summary should be updated automatically
    assert_eq!(composite.resource_summary.total_qubits_used, 4); // qubits 0, 1, 2, 3
    assert_eq!(composite.resource_summary.total_classical_used, 2); // classical 0, 1
}

#[test] 
fn test_swap_operation_routing() {
    let topology = TopologyBuilder::linear(5);
    let mapper = QubitMapper::new(&topology);
    
    let mut mapping = vec![0, 4]; // Logical 0->physical 0, logical 1->physical 4
    
    // Try to route a two-qubit gate between qubits 0 and 4 (not directly connected)
    let swaps = mapper.route_two_qubit_gate(0, 1, &mut mapping).unwrap();
    
    // Should need SWAP operations to bring the qubits together
    assert!(!swaps.is_empty());
    
    // Each swap should be between adjacent qubits in the linear topology
    for swap in &swaps {
        assert!(topology.are_connected(Qubit(swap.qubit1), Qubit(swap.qubit2)));
    }
}

#[tokio::test]
async fn test_end_to_end_scheduling_and_composition() {
    let topology = TopologyBuilder::grid(3, 3);
    let mut qvm = QvmScheduler::new(topology);
    
    // Create test circuits
    let mut circuit1 = QuantumCircuit::new("bell_state".to_string(), 2, 2);
    circuit1.add_operation(Operation::SingleQubit {
        gate: SingleQubitGate::H,
        qubit: Qubit(0),
        parameters: SmallVec::new(),
    });
    circuit1.add_operation(Operation::TwoQubit {
        gate: TwoQubitGate::CNOT,
        control: Qubit(0),
        target: Qubit(1),
        parameters: SmallVec::new(),
    });
    circuit1.add_operation(Operation::Measurement {
        qubit: Qubit(0),
        classical: ClassicalBit(0),
    });
    circuit1.add_operation(Operation::Measurement {
        qubit: Qubit(1),
        classical: ClassicalBit(1),
    });

    let mut circuit2 = QuantumCircuit::new("simple_x".to_string(), 1, 1);
    circuit2.add_operation(Operation::SingleQubit {
        gate: SingleQubitGate::X,
        qubit: Qubit(0),
        parameters: SmallVec::new(),
    });
    circuit2.add_operation(Operation::Measurement {
        qubit: Qubit(0),
        classical: ClassicalBit(0),
    });

    // Schedule and compose
    let composite = qvm.schedule_circuits(vec![circuit1, circuit2]).await.unwrap();
    
    // Verify composition
    assert_eq!(composite.circuit_count(), 2);
    assert!(composite.total_duration() > 0);
    
    // Generate QASM output
    let qasm_output = composite.to_qasm().unwrap();
    assert!(qasm_output.contains("OPENQASM 3.0"));
    assert!(qasm_output.contains("h q"));
    assert!(qasm_output.contains("cx q"));
    assert!(qasm_output.contains("x q"));
    assert!(qasm_output.contains("measure q"));
    
    // Verify that physical qubits are used (not just logical)
    assert!(qasm_output.contains("qubit["));
    assert!(qasm_output.contains("bit["));
}