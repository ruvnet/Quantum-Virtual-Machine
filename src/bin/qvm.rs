//! QVM CLI - Command-line interface for the Quantum Virtual Machine scheduler

use qvm_scheduler::{QvmScheduler, CircuitBuilder, TopologyBuilder};
use std::env;
use std::process;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_help();
        process::exit(0);
    }

    match args[1].as_str() {
        "demo" => run_demo().await,
        "schedule" => {
            if args.len() < 3 {
                eprintln!("Error: Please specify number of circuits to schedule");
                process::exit(1);
            }
            let num_circuits = args[2].parse().unwrap_or(2);
            run_schedule(num_circuits).await;
        }
        "help" | "--help" | "-h" => print_help(),
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_help();
            process::exit(1);
        }
    }
}

fn print_help() {
    println!("QVM Scheduler - Quantum Virtual Machine CLI");
    println!("\nUsage: qvm <command> [options]");
    println!("\nCommands:");
    println!("  demo              Run a demonstration with Bell and GHZ states");
    println!("  schedule <n>      Schedule n random circuits");
    println!("  help              Show this help message");
}

async fn run_demo() {
    println!("🚀 QVM Scheduler Demo");
    println!("=====================\n");

    // Create topology
    let topology = TopologyBuilder::grid(5, 5);
    println!("✅ Created 5x5 grid topology (25 qubits)");

    // Initialize scheduler
    let scheduler = QvmScheduler::new(topology);
    println!("✅ Initialized QVM scheduler");

    // Create circuits
    let bell = CircuitBuilder::new("bell_state", 2, 2)
        .h(0).unwrap()
        .cx(0, 1).unwrap()
        .measure_all().unwrap()
        .build();
    println!("✅ Created Bell state circuit (2 qubits)");

    let ghz = CircuitBuilder::new("ghz_state", 3, 3)
        .h(0).unwrap()
        .cx(0, 1).unwrap()
        .cx(1, 2).unwrap()
        .measure_all().unwrap()
        .build();
    println!("✅ Created GHZ state circuit (3 qubits)");

    // Schedule circuits
    let circuits = vec![bell, ghz];
    println!("\n📊 Scheduling {} circuits...", circuits.len());
    
    match scheduler.schedule(&circuits).await {
        Ok(composite) => {
            println!("✅ Scheduling successful!");
            println!("\nSchedule Summary:");
            println!("  Total circuits: {}", composite.circuits().len());
            println!("  Total duration: {} μs", composite.total_duration());
            println!("  Total qubits used: {}", composite.total_qubits());
            println!("  Total classical bits: {}", composite.total_cbits());
            
            // Generate QASM
            match composite.to_qasm() {
                Ok(qasm) => {
                    println!("\n📝 Generated OpenQASM 3.0 output:");
                    println!("------------------------------------");
                    // Show first 500 chars of QASM
                    if qasm.len() > 500 {
                        println!("{}...\n[Output truncated]", &qasm[..500]);
                    } else {
                        println!("{}", qasm);
                    }
                },
                Err(e) => eprintln!("❌ Failed to generate QASM: {}", e),
            }
        },
        Err(e) => eprintln!("❌ Scheduling failed: {}", e),
    }
}

async fn run_schedule(num_circuits: usize) {
    println!("🚀 Scheduling {} random circuits", num_circuits);
    
    // Create topology
    let topology = TopologyBuilder::grid(10, 10);
    let scheduler = QvmScheduler::new(topology);
    
    // Generate random circuits
    let mut circuits = Vec::new();
    for i in 0..num_circuits {
        let qubits = 2 + (i % 4); // 2-5 qubits
        let circuit = CircuitBuilder::new(&format!("circuit_{}", i), qubits, qubits)
            .h(0).unwrap()
            .cx(0, 1 % qubits).unwrap()
            .measure_all().unwrap()
            .build();
        circuits.push(circuit);
    }
    
    println!("✅ Generated {} circuits", circuits.len());
    println!("📊 Scheduling...");
    
    let start = std::time::Instant::now();
    match scheduler.schedule(&circuits).await {
        Ok(composite) => {
            let duration = start.elapsed();
            println!("\n✅ Scheduling completed in {:.2?}", duration);
            println!("\nResults:");
            println!("  Circuits scheduled: {}", composite.circuits().len());
            println!("  Total duration: {} μs", composite.total_duration());
            println!("  Qubits used: {}/{}", composite.total_qubits(), 100);
            println!("  Utilization: {:.1}%", (composite.total_qubits() as f64 / 100.0) * 100.0);
        },
        Err(e) => eprintln!("❌ Scheduling failed: {}", e),
    }
}