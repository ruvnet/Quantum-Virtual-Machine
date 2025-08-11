//! Comprehensive benchmarks for the quantum circuit scheduler

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use qvm_scheduler::*;
use qvm_scheduler::scheduler::*;
use qvm_scheduler::topology::TopologyBuilder;
use qvm_scheduler::circuit_ir::CircuitBuilder;
use std::time::Duration;
use tokio::runtime::Runtime;

/// Benchmark configuration
struct BenchConfig {
    topology_size: (usize, usize),
    job_count: usize,
    qubits_per_job: usize,
}

impl BenchConfig {
    fn new(topology_size: (usize, usize), job_count: usize, qubits_per_job: usize) -> Self {
        Self {
            topology_size,
            job_count,
            qubits_per_job,
        }
    }
}

/// Generate test jobs for benchmarking
fn generate_test_jobs(count: usize, qubits_per_job: usize) -> Vec<Job> {
    let mut jobs = Vec::new();
    
    for i in 0..count {
        let circuit = CircuitBuilder::new(&format!("bench_circuit_{}", i), qubits_per_job, qubits_per_job)
            .h(0)
            .unwrap()
            .build();
        
        let mut job = Job::new(i, circuit);
        job.estimated_duration = 1000 + (i as u64 * 100); // Variable duration
        job.priority = (i % 3) as i32; // Variable priority
        jobs.push(job);
    }
    
    jobs
}

/// Benchmark bin-packing algorithms
fn bench_bin_packing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let configs = [
        BenchConfig::new((4, 4), 10, 2),
        BenchConfig::new((6, 6), 20, 3),
        BenchConfig::new((8, 8), 50, 4),
    ];

    let algorithms = [
        BinPackingAlgorithm::FirstFitDecreasing,
        BinPackingAlgorithm::BestFitDecreasing,
        BinPackingAlgorithm::WorstFitDecreasing,
        BinPackingAlgorithm::NextFit,
    ];

    let mut group = c.benchmark_group("bin_packing");
    
    for config in &configs {
        let topology = TopologyBuilder::grid(config.topology_size.0, config.topology_size.1);
        let jobs = generate_test_jobs(config.job_count, config.qubits_per_job);
        
        for &algorithm in &algorithms {
            let bench_id = BenchmarkId::from_parameter(format!(
                "{:?}_{}x{}_{}jobs_{}qubits",
                algorithm,
                config.topology_size.0,
                config.topology_size.1,
                config.job_count,
                config.qubits_per_job
            ));
            
            group.throughput(Throughput::Elements(config.job_count as u64));
            group.bench_with_input(bench_id, &(topology.clone(), jobs.clone(), algorithm), |b, (topo, jobs, algo)| {
                b.to_async(&rt).iter(|| async {
                    let bin_packer = BinPacker::with_algorithm(*algo);
                    let batch = Batch {
                        id: 0,
                        jobs: jobs.clone(),
                        metadata: Default::default(),
                    };
                    let _result = black_box(bin_packer.pack_batch(batch, topo).await.unwrap());
                });
            });
        }
    }
    
    group.finish();
}

/// Benchmark batch scheduling
fn bench_batch_scheduling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let configs = [
        BenchConfig::new((6, 6), 30, 2),
        BenchConfig::new((8, 8), 60, 3),
        BenchConfig::new((10, 10), 100, 4),
    ];

    let strategies = [
        BatchingStrategy::FirstComeFirstServed,
        BatchingStrategy::ShortestJobFirst,
        BatchingStrategy::LongestJobFirst,
        BatchingStrategy::PriorityBased,
    ];

    let mut group = c.benchmark_group("batch_scheduling");
    
    for config in &configs {
        let topology = TopologyBuilder::grid(config.topology_size.0, config.topology_size.1);
        let jobs = generate_test_jobs(config.job_count, config.qubits_per_job);
        
        for &strategy in &strategies {
            let batch_config = BatchConfig {
                strategy,
                max_batch_size: 10,
                resource_threshold: 0.8,
                ..Default::default()
            };
            
            let bench_id = BenchmarkId::from_parameter(format!(
                "{:?}_{}x{}_{}jobs",
                strategy,
                config.topology_size.0,
                config.topology_size.1,
                config.job_count
            ));
            
            group.throughput(Throughput::Elements(config.job_count as u64));
            group.bench_with_input(bench_id, &(topology.clone(), jobs.clone(), batch_config), |b, (topo, jobs, config)| {
                b.to_async(&rt).iter(|| async {
                    let scheduler = BatchScheduler::new(config.clone());
                    let _result = black_box(scheduler.create_batches(jobs.clone(), config.clone()).await.unwrap());
                });
            });
        }
    }
    
    group.finish();
}

/// Benchmark memory and scalability
fn bench_scalability(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    // Test with increasing job counts
    let job_counts = [10, 50, 100, 200];
    let topology = TopologyBuilder::grid(10, 10); // Large topology
    
    let mut group = c.benchmark_group("scalability");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    
    for &job_count in &job_counts {
        let jobs = generate_test_jobs(job_count, 3);
        
        let bench_id = BenchmarkId::from_parameter(format!("{}jobs", job_count));
        group.throughput(Throughput::Elements(job_count as u64));
        group.bench_with_input(bench_id, &(topology.clone(), jobs), |b, (topo, jobs)| {
            b.to_async(&rt).iter(|| async {
                let bin_packer = BinPacker::with_algorithm(BinPackingAlgorithm::FirstFitDecreasing);
                let batch = Batch {
                    id: 0,
                    jobs: jobs.clone(),
                    metadata: Default::default(),
                };
                let _result = black_box(bin_packer.pack_batch(batch, topo).await.unwrap());
            });
        });
    }
    
    group.finish();
}

/// Benchmark topology effects
fn bench_topology_effects(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let topologies = [
        ("grid_4x4", TopologyBuilder::grid(4, 4)),
        ("grid_8x8", TopologyBuilder::grid(8, 8)),
        ("linear_16", TopologyBuilder::linear(16)),
        ("linear_64", TopologyBuilder::linear(64)),
        ("ring_20", TopologyBuilder::ring(20)),
        ("star_15", TopologyBuilder::star(15)),
    ];
    
    let jobs = generate_test_jobs(20, 3);
    
    let mut group = c.benchmark_group("topology_effects");
    
    for (name, topology) in &topologies {
        let bench_id = BenchmarkId::from_parameter(name);
        group.throughput(Throughput::Elements(20));
        group.bench_with_input(bench_id, &(topology.clone(), jobs.clone()), |b, (topo, jobs)| {
            b.to_async(&rt).iter(|| async {
                let bin_packer = BinPacker::with_algorithm(BinPackingAlgorithm::FirstFitDecreasing);
                let batch = Batch {
                    id: 0,
                    jobs: jobs.clone(),
                    metadata: Default::default(),
                };
                let _result = black_box(bin_packer.pack_batch(batch, topo).await.unwrap());
            });
        });
    }
    
    group.finish();
}

/// Benchmark complete scheduler integration
fn bench_full_scheduler(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let configs = [
        BenchConfig::new((6, 6), 25, 3),
        BenchConfig::new((8, 8), 50, 4),
    ];
    
    let mut group = c.benchmark_group("full_scheduler");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(30));
    
    for config in &configs {
        let topology = TopologyBuilder::grid(config.topology_size.0, config.topology_size.1);
        
        // Create diverse job mix
        let mut jobs = Vec::new();
        for i in 0..config.job_count {
            let qubits = 2 + (i % 3); // Variable qubit requirements
            let circuit = CircuitBuilder::new(&format!("circuit_{}", i), qubits, qubits)
                .h(0)
                .unwrap()
                .build();
            
            let mut job = Job::new(i, circuit);
            job.priority = (i % 5) as i32;
            job.estimated_duration = 500 + (i as u64 * 50);
            jobs.push(job);
        }
        
        let bench_id = BenchmarkId::from_parameter(format!(
            "{}x{}_{}jobs",
            config.topology_size.0,
            config.topology_size.1,
            config.job_count
        ));
        
        group.throughput(Throughput::Elements(config.job_count as u64));
        group.bench_with_input(bench_id, &(topology.clone(), jobs), |b, (topo, jobs)| {
            b.to_async(&rt).iter(|| async {
                let mut scheduler = Scheduler::new(topo.clone());
                let _result = black_box(scheduler.schedule(jobs.clone()).await.unwrap());
            });
        });
    }
    
    group.finish();
}

/// Original scheduling benchmark (maintained for backwards compatibility)
fn bench_scheduling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("qvm_scheduling");
    
    for circuit_count in [1, 5, 10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("schedule_circuits", circuit_count),
            circuit_count,
            |b, &circuit_count| {
                let topology = TopologyBuilder::grid(5, 5);
                let circuits: Vec<QuantumCircuit> = (0..circuit_count)
                    .map(|i| {
                        CircuitBuilder::new(&format!("circuit_{}", i), 3, 3)
                            .h(0).unwrap()
                            .measure_all().unwrap()
                            .build()
                    })
                    .collect();
                
                b.iter(|| {
                    rt.block_on(async {
                        let mut scheduler = QvmScheduler::new(topology.clone());
                        let result = scheduler.schedule_circuits(black_box(circuits.clone())).await;
                        black_box(result.unwrap())
                    })
                });
            },
        );
    }
    
    group.finish();
}

/// Topology creation benchmark (maintained for backwards compatibility)
fn bench_topology_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_creation");
    
    for size in [3, 5, 10, 20].iter() {
        group.bench_with_input(
            BenchmarkId::new("grid_topology", size),
            size,
            |b, &size| {
                b.iter(|| {
                    black_box(TopologyBuilder::grid(size, size))
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_bin_packing,
    bench_batch_scheduling,
    bench_scalability,
    bench_topology_effects,
    bench_full_scheduler,
    bench_scheduling,
    bench_topology_creation
);

criterion_main!(benches);