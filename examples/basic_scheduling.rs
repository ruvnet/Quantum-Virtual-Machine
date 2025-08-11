//! Basic scheduling example

use qvm_scheduler::{QvmScheduler, QuantumCircuit};
use qvm_scheduler::topology::TopologyBuilder;
use qvm_scheduler::circuit_ir::CircuitBuilder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(\"QVM Scheduler Basic Example\");

    // Create a 3x3 grid topology
    let topology = TopologyBuilder::grid(3, 3);
    println!(\"Created {}x{} grid topology with {} qubits\", 
             3, 3, topology.qubit_count());

    // Create the scheduler
    let mut scheduler = QvmScheduler::new(topology);

    // Create some example circuits
    let circuit1 = CircuitBuilder::new(\"bell_state\", 2, 2)
        .bell_state(0, 1)?
        .measure_all()?
        .build();

    let circuit2 = CircuitBuilder::new(\"ghz_state\", 3, 3)
        .ghz_state(&[0.into(), 1.into(), 2.into()])?
        .measure_all()?
        .build();

    let circuits = vec![circuit1, circuit2];
    println!(\"Created {} circuits to schedule\", circuits.len());

    // Schedule the circuits
    let composite = scheduler.schedule_circuits(circuits).await?;
    println!(\"Scheduled {} circuits successfully\", composite.circuits().len());

    // Generate QASM output
    let qasm_output = composite.to_qasm()?;
    println!(\"\\nGenerated QASM:\\n{}\", qasm_output);

    Ok(())
}