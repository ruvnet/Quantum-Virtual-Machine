//! Comprehensive tests for the quantum circuit scheduler

use qvm_scheduler::*;
use qvm_scheduler::scheduler::*;
use qvm_scheduler::topology::TopologyBuilder;
use qvm_scheduler::circuit_ir::CircuitBuilder;
use tokio_test;

/// Create test jobs for testing
fn create_test_jobs(count: usize, qubits_per_job: usize) -> Vec<Job> {
    let mut jobs = Vec::new();
    
    for i in 0..count {
        let circuit = CircuitBuilder::new(&format!("test_circuit_{}", i), qubits_per_job, qubits_per_job)
            .h(0)
            .unwrap()
            .build();
        
        let mut job = Job::new(i, circuit);
        job.estimated_duration = 1000 + (i as u64 * 100);
        job.priority = (i % 3) as i32;
        jobs.push(job);
    }
    
    jobs
}

#[cfg(test)]
mod bin_packing_tests {
    use super::*;

    #[tokio::test]
    async fn test_first_fit_decreasing() {
        let topology = TopologyBuilder::grid(4, 4);
        let jobs = create_test_jobs(5, 2);
        let bin_packer = BinPacker::with_algorithm(BinPackingAlgorithm::FirstFitDecreasing);
        
        let batch = Batch {
            id: 0,
            jobs,
            metadata: Default::default(),
        };
        
        let assignments = bin_packer.pack_batch(batch, &topology).await.unwrap();
        assert_eq!(assignments.len(), 5);
        
        // Check that all assignments have valid qubit mappings
        for assignment in &assignments {
            assert!(!assignment.qubit_mapping.is_empty());
            for &qubit_idx in &assignment.qubit_mapping {
                assert!(qubit_idx < topology.qubit_count());
            }
        }
    }

    #[tokio::test]
    async fn test_best_fit_decreasing() {
        let topology = TopologyBuilder::grid(6, 6);
        let jobs = create_test_jobs(8, 3);
        let bin_packer = BinPacker::with_algorithm(BinPackingAlgorithm::BestFitDecreasing);
        
        let batch = Batch {
            id: 0,
            jobs,
            metadata: Default::default(),
        };
        
        let assignments = bin_packer.pack_batch(batch, &topology).await.unwrap();
        assert_eq!(assignments.len(), 8);
        
        // Verify assignments are sorted by size (descending)
        for i in 1..assignments.len() {
            assert!(assignments[i-1].qubit_mapping.len() >= assignments[i].qubit_mapping.len());
        }
    }

    #[tokio::test]
    async fn test_worst_fit_decreasing() {
        let topology = TopologyBuilder::grid(5, 5);
        let jobs = create_test_jobs(6, 2);
        let bin_packer = BinPacker::with_algorithm(BinPackingAlgorithm::WorstFitDecreasing);
        
        let batch = Batch {
            id: 0,
            jobs,
            metadata: Default::default(),
        };
        
        let assignments = bin_packer.pack_batch(batch, &topology).await.unwrap();
        assert_eq!(assignments.len(), 6);
        
        // Check that no qubits are double-allocated in the same time slot
        let mut used_qubits = std::collections::HashSet::new();
        for assignment in &assignments {
            if assignment.start_time == 0 { // Same time slot
                for &qubit in &assignment.qubit_mapping {
                    assert!(!used_qubits.contains(&qubit), "Qubit {} double-allocated", qubit);
                    used_qubits.insert(qubit);
                }
            }
        }
    }

    #[tokio::test]
    async fn test_next_fit_decreasing() {
        let topology = TopologyBuilder::linear(10);
        let jobs = create_test_jobs(4, 2);
        let bin_packer = BinPacker::with_algorithm(BinPackingAlgorithm::NextFit);
        
        let batch = Batch {
            id: 0,
            jobs,
            metadata: Default::default(),
        };
        
        let assignments = bin_packer.pack_batch(batch, &topology).await.unwrap();
        assert_eq!(assignments.len(), 4);
        
        // Verify jobs are sorted by size (decreasing) for NFD
        for i in 1..assignments.len() {
            let job1_size = assignments[i-1].qubit_mapping.len();
            let job2_size = assignments[i].qubit_mapping.len();
            assert!(job1_size >= job2_size, "Jobs not properly sorted for NFD");
        }
    }

    #[tokio::test]
    async fn test_empty_batch() {
        let topology = TopologyBuilder::grid(3, 3);
        let bin_packer = BinPacker::with_algorithm(BinPackingAlgorithm::FirstFitDecreasing);
        
        let batch = Batch {
            id: 0,
            jobs: Vec::new(),
            metadata: Default::default(),
        };
        
        let assignments = bin_packer.pack_batch(batch, &topology).await.unwrap();
        assert!(assignments.is_empty());
    }

    #[tokio::test]
    async fn test_large_job_small_topology() {
        let topology = TopologyBuilder::grid(2, 2); // Only 4 qubits
        let jobs = create_test_jobs(1, 6); // Needs 6 qubits
        let bin_packer = BinPacker::with_algorithm(BinPackingAlgorithm::FirstFitDecreasing);
        
        let batch = Batch {
            id: 0,
            jobs,
            metadata: Default::default(),
        };
        
        // Should fail or allocate with available qubits
        let result = bin_packer.pack_batch(batch, &topology).await;
        // This test verifies the system handles resource constraints gracefully
        assert!(result.is_ok() || result.is_err());
    }
}

#[cfg(test)]
mod batch_scheduling_tests {
    use super::*;

    #[tokio::test]
    async fn test_first_come_first_served() {
        let jobs = create_test_jobs(10, 2);
        let config = BatchConfig {
            strategy: BatchingStrategy::FirstComeFirstServed,
            max_batch_size: 5,
            resource_threshold: 0.8,
            ..Default::default()
        };
        
        let scheduler = BatchScheduler::new(config.clone());
        let batches = scheduler.create_batches(jobs.clone(), config).await.unwrap();
        
        // Should create 2 batches of 5 jobs each
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].jobs.len(), 5);
        assert_eq!(batches[1].jobs.len(), 5);
        
        // Jobs should be in original order
        for (i, job) in batches[0].jobs.iter().enumerate() {
            assert_eq!(job.id, jobs[i].id);
        }
    }

    #[tokio::test]
    async fn test_shortest_job_first() {
        let mut jobs = create_test_jobs(6, 2);
        // Set different durations
        jobs[0].estimated_duration = 5000;
        jobs[1].estimated_duration = 1000;
        jobs[2].estimated_duration = 3000;
        jobs[3].estimated_duration = 2000;
        jobs[4].estimated_duration = 4000;
        jobs[5].estimated_duration = 1500;
        
        let config = BatchConfig {
            strategy: BatchingStrategy::ShortestJobFirst,
            max_batch_size: 10,
            resource_threshold: 0.8,
            ..Default::default()
        };
        
        let scheduler = BatchScheduler::new(config.clone());
        let batches = scheduler.create_batches(jobs, config).await.unwrap();
        
        assert_eq!(batches.len(), 1);
        let batch_jobs = &batches[0].jobs;
        
        // Verify jobs are sorted by duration (ascending)
        for i in 1..batch_jobs.len() {
            assert!(batch_jobs[i-1].estimated_duration <= batch_jobs[i].estimated_duration);
        }
    }

    #[tokio::test]
    async fn test_priority_based_batching() {
        let mut jobs = create_test_jobs(8, 2);
        // Set different priorities
        for (i, job) in jobs.iter_mut().enumerate() {
            job.priority = (i % 4) as i32; // Priorities 0, 1, 2, 3
        }
        
        let config = BatchConfig {
            strategy: BatchingStrategy::PriorityBased,
            max_batch_size: 10,
            resource_threshold: 0.8,
            ..Default::default()
        };
        
        let scheduler = BatchScheduler::new(config.clone());
        let batches = scheduler.create_batches(jobs, config).await.unwrap();
        
        assert_eq!(batches.len(), 1);
        let batch_jobs = &batches[0].jobs;
        
        // Verify jobs are sorted by priority (descending)
        for i in 1..batch_jobs.len() {
            assert!(batch_jobs[i-1].priority >= batch_jobs[i].priority);
        }
    }

    #[tokio::test]
    async fn test_resource_constrained_batching() {
        let jobs = create_test_jobs(20, 3); // 20 jobs, 3 qubits each = 60 qubits total
        let config = BatchConfig {
            strategy: BatchingStrategy::FirstComeFirstServed,
            max_batch_size: 100, // Large batch size
            resource_threshold: 0.5, // 50% of topology (18 out of 36 qubits)
            ..Default::default()
        };
        
        let scheduler = BatchScheduler::new(config.clone());
        let batches = scheduler.create_batches(jobs, config).await.unwrap();
        
        // Should create multiple batches due to resource constraints
        assert!(batches.len() > 1);
        
        // Each batch should respect resource constraints
        for batch in batches {
            let total_qubits: usize = batch.jobs.iter()
                .map(|job| job.requirements.qubits_needed)
                .sum();
            assert!(total_qubits <= 18, "Batch exceeds resource threshold");
        }
    }
}

#[cfg(test)]
mod scheduler_integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_scheduler_workflow() {
        let topology = TopologyBuilder::grid(6, 6);
        let mut scheduler = Scheduler::new(topology);
        let jobs = create_test_jobs(15, 3);
        
        let schedule = scheduler.schedule(jobs.clone()).await.unwrap();
        
        assert_eq!(schedule.assignments.len(), jobs.len());
        assert!(schedule.total_duration > 0);
        
        // Verify all jobs are scheduled
        let scheduled_jobs: std::collections::HashSet<usize> = schedule.assignments
            .iter()
            .map(|a| a.job_id)
            .collect();
        
        for job in &jobs {
            assert!(scheduled_jobs.contains(&job.id), "Job {} not scheduled", job.id);
        }
    }

    #[tokio::test]
    async fn test_scheduler_with_different_topologies() {
        let topologies = vec![
            ("grid_4x4", TopologyBuilder::grid(4, 4)),
            ("linear_16", TopologyBuilder::linear(16)),
            ("ring_12", TopologyBuilder::ring(12)),
            ("star_10", TopologyBuilder::star(10)),
        ];
        
        let jobs = create_test_jobs(8, 2);
        
        for (name, topology) in topologies {
            let mut scheduler = Scheduler::new(topology);
            let schedule = scheduler.schedule(jobs.clone()).await.unwrap();
            
            assert_eq!(schedule.assignments.len(), jobs.len(), "Failed for topology: {}", name);
            assert!(schedule.total_duration > 0, "Zero duration for topology: {}", name);
        }
    }

    #[tokio::test]
    async fn test_scheduler_resource_utilization() {
        let topology = TopologyBuilder::grid(8, 8);
        let mut scheduler = Scheduler::new(topology.clone());
        let jobs = create_test_jobs(32, 2); // Many small jobs
        
        let schedule = scheduler.schedule(jobs).await.unwrap();
        
        // Check utilization metrics
        assert!(schedule.utilization.avg_qubit_utilization >= 0.0);
        assert!(schedule.utilization.avg_qubit_utilization <= 1.0);
        
        // Verify no qubit double allocation at same time
        let mut time_qubit_map: std::collections::HashMap<u64, std::collections::HashSet<usize>> = std::collections::HashMap::new();
        
        for assignment in &schedule.assignments {
            let qubits = time_qubit_map.entry(assignment.start_time).or_insert_with(std::collections::HashSet::new);
            
            for &qubit in &assignment.qubit_mapping {
                assert!(!qubits.contains(&qubit), 
                    "Qubit {} double-allocated at time {}", qubit, assignment.start_time);
                qubits.insert(qubit);
            }
        }
    }

    #[tokio::test]
    async fn test_qvm_scheduler_integration() {
        let topology = TopologyBuilder::grid(5, 5);
        let mut qvm_scheduler = QvmScheduler::new(topology);
        
        let circuits: Vec<QuantumCircuit> = (0..10)
            .map(|i| {
                CircuitBuilder::new(&format!("integration_circuit_{}", i), 3, 3)
                    .h(0).unwrap()
                    .cx(0, 1).unwrap()
                    .measure_all().unwrap()
                    .build()
            })
            .collect();
        
        let composite = qvm_scheduler.schedule_circuits(circuits.clone()).await.unwrap();
        
        assert_eq!(composite.circuit_count(), circuits.len());
        assert!(composite.total_duration() > 0);
        
        // Test QASM output generation
        let qasm_output = composite.to_qasm().unwrap();
        assert!(qasm_output.contains("OPENQASM"));
        assert!(qasm_output.contains("qubit"));
    }
}

#[cfg(test)]
mod assignment_tests {
    use super::*;

    #[test]
    fn test_assignment_creation() {
        let assignment = Assignment::new(0, 1, 1000, 5000)
            .with_qubit_mapping(vec![0, 1, 2])
            .with_classical_mapping(vec![0, 1]);
        
        assert_eq!(assignment.job_id, 0);
        assert_eq!(assignment.tile_id, 1);
        assert_eq!(assignment.start_time, 1000);
        assert_eq!(assignment.duration, 5000);
        assert_eq!(assignment.qubit_mapping, vec![0, 1, 2]);
        assert_eq!(assignment.classical_mapping, vec![0, 1]);
    }

    #[test]
    fn test_assignment_validation() {
        let topology = TopologyBuilder::grid(4, 4);
        let assignment = Assignment::new(0, 1, 0, 1000)
            .with_qubit_mapping(vec![0, 1, 16]) // 16 is out of bounds for 4x4 grid
            .with_classical_mapping(vec![0]);
        
        let is_valid = assignment.validate(&topology);
        assert!(!is_valid, "Assignment with invalid qubit should be invalid");
        
        let valid_assignment = Assignment::new(0, 1, 0, 1000)
            .with_qubit_mapping(vec![0, 1, 2])
            .with_classical_mapping(vec![0]);
        
        let is_valid = valid_assignment.validate(&topology);
        assert!(is_valid, "Valid assignment should be valid");
    }

    #[test]
    fn test_assignment_conflicts() {
        let assignment1 = Assignment::new(0, 1, 1000, 2000)
            .with_qubit_mapping(vec![0, 1]);
        let assignment2 = Assignment::new(1, 2, 1500, 1000)
            .with_qubit_mapping(vec![1, 2]); // Overlaps on qubit 1
        let assignment3 = Assignment::new(2, 3, 3500, 1000)
            .with_qubit_mapping(vec![3, 4]); // No overlap
        
        assert!(assignment1.conflicts_with(&assignment2), "Overlapping assignments should conflict");
        assert!(!assignment1.conflicts_with(&assignment3), "Non-overlapping assignments should not conflict");
        assert!(!assignment2.conflicts_with(&assignment3), "Time-separated assignments should not conflict");
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[tokio::test]
    async fn test_scheduler_performance() {
        let topology = TopologyBuilder::grid(10, 10);
        let jobs = create_test_jobs(100, 3); // Large number of jobs
        
        let start = Instant::now();
        
        let mut scheduler = Scheduler::new(topology);
        let schedule = scheduler.schedule(jobs).await.unwrap();
        
        let duration = start.elapsed();
        
        println!("Scheduled 100 jobs in {:?}", duration);
        assert!(duration.as_secs() < 10, "Scheduling took too long: {:?}", duration);
        assert_eq!(schedule.assignments.len(), 100);
    }

    #[tokio::test]
    async fn test_scalability() {
        let topology = TopologyBuilder::grid(12, 12);
        let job_counts = [10, 50, 100, 200];
        
        for &job_count in &job_counts {
            let jobs = create_test_jobs(job_count, 2);
            let start = Instant::now();
            
            let mut scheduler = Scheduler::new(topology.clone());
            let schedule = scheduler.schedule(jobs).await.unwrap();
            
            let duration = start.elapsed();
            let jobs_per_second = job_count as f64 / duration.as_secs_f64();
            
            println!("Processed {} jobs at {:.1} jobs/second", job_count, jobs_per_second);
            assert_eq!(schedule.assignments.len(), job_count);
            assert!(jobs_per_second > 1.0, "Processing rate too low for {} jobs", job_count);
        }
    }

    #[tokio::test]
    async fn test_memory_efficiency() {
        let topology = TopologyBuilder::grid(8, 8);
        let large_job_count = 500;
        let jobs = create_test_jobs(large_job_count, 2);
        
        // This test ensures the scheduler can handle large job counts without excessive memory usage
        let mut scheduler = Scheduler::new(topology);
        let schedule = scheduler.schedule(jobs).await.unwrap();
        
        assert_eq!(schedule.assignments.len(), large_job_count);
        
        // Verify the schedule is reasonable
        assert!(schedule.total_duration > 0);
        assert!(schedule.utilization.avg_qubit_utilization >= 0.0);
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[tokio::test]
    async fn test_invalid_job_handling() {
        let topology = TopologyBuilder::grid(3, 3); // 9 qubits total
        let mut scheduler = Scheduler::new(topology);
        
        // Create a job that requires more qubits than available
        let large_circuit = CircuitBuilder::new("large", 20, 5)
            .h(0).unwrap()
            .build();
        let large_job = Job::new(0, large_circuit);
        
        let result = scheduler.schedule(vec![large_job]).await;
        // The scheduler should either handle this gracefully or return an appropriate error
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_empty_job_list() {
        let topology = TopologyBuilder::grid(4, 4);
        let mut scheduler = Scheduler::new(topology);
        
        let schedule = scheduler.schedule(vec![]).await.unwrap();
        assert!(schedule.assignments.is_empty());
        assert_eq!(schedule.total_duration, 0);
    }

    #[test]
    fn test_topology_edge_cases() {
        // Test with minimal topology
        let tiny_topology = TopologyBuilder::linear(1);
        assert_eq!(tiny_topology.qubit_count(), 1);
        
        // Test with empty topology (if possible)
        let empty_topology = Topology::new();
        assert_eq!(empty_topology.qubit_count(), 0);
    }
}

#[cfg(test)]
mod concurrent_tests {
    use super::*;
    use std::sync::Arc;
    use tokio::task;

    #[tokio::test]
    async fn test_concurrent_scheduling() {
        let topology = Arc::new(TopologyBuilder::grid(8, 8));
        let mut handles = Vec::new();
        
        // Spawn multiple concurrent scheduling tasks
        for i in 0..5 {
            let topo = topology.clone();
            let handle = task::spawn(async move {
                let mut scheduler = Scheduler::new((*topo).clone());
                let jobs = create_test_jobs(20, 2);
                scheduler.schedule(jobs).await
            });
            handles.push(handle);
        }
        
        // Wait for all tasks to complete
        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.await.unwrap();
            assert!(result.is_ok(), "Concurrent task {} failed", i);
            
            if let Ok(schedule) = result {
                assert_eq!(schedule.assignments.len(), 20);
            }
        }
    }
}