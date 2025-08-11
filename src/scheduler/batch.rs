//! Batch scheduling for grouping related jobs

use crate::{QvmError, Result, Topology};
use crate::scheduler::Job;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Batch of jobs to be scheduled together
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Batch {
    /// Batch identifier
    pub id: usize,
    /// Jobs in this batch
    pub jobs: Vec<Job>,
    /// Batch metadata
    pub metadata: BatchMetadata,
}

/// Batch metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchMetadata {
    /// Batch creation timestamp
    pub created_at: u64,
    /// Batch priority
    pub priority: i32,
    /// Expected batch duration
    pub estimated_duration: u64,
    /// Batch type
    pub batch_type: BatchType,
    /// Custom properties
    pub properties: HashMap<String, String>,
}

/// Types of batches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchType {
    /// Regular batch of independent jobs
    Independent,
    /// Jobs that should run in parallel
    Parallel,
    /// Jobs that must run sequentially
    Sequential,
    /// Jobs with shared resources
    SharedResource,
    /// High-priority batch
    HighPriority,
}

impl Default for BatchType {
    fn default() -> Self {
        Self::Independent
    }
}

/// Configuration for batch creation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchConfig {
    /// Maximum jobs per batch
    pub max_jobs_per_batch: usize,
    /// Maximum batch duration
    pub max_batch_duration: u64,
    /// Batching strategy
    pub strategy: BatchingStrategy,
    /// Enable resource-based batching
    pub group_by_resources: bool,
    /// Enable priority-based batching
    pub group_by_priority: bool,
    /// Enable deadline-aware batching
    pub deadline_aware: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_jobs_per_batch: 10,
            max_batch_duration: 1_000_000, // 1 second
            strategy: BatchingStrategy::ResourceAware,
            group_by_resources: true,
            group_by_priority: true,
            deadline_aware: true,
        }
    }
}

/// Batching strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BatchingStrategy {
    /// Simple first-come-first-served
    FirstComeFirstServed,
    /// Group by resource requirements
    ResourceAware,
    /// Group by priority levels
    PriorityBased,
    /// Group by deadlines
    DeadlineAware,
    /// Adaptive strategy
    Adaptive,
}

impl Batch {
    /// Create a new batch
    pub fn new(id: usize) -> Self {
        Self {
            id,
            jobs: Vec::new(),
            metadata: BatchMetadata::default(),
        }
    }

    /// Add a job to the batch
    pub fn add_job(&mut self, job: Job) -> Result<()> {
        self.jobs.push(job);
        self.update_metadata();
        Ok(())
    }

    /// Remove a job from the batch
    pub fn remove_job(&mut self, job_id: usize) -> Option<Job> {
        if let Some(pos) = self.jobs.iter().position(|j| j.id == job_id) {
            let removed = self.jobs.remove(pos);
            self.update_metadata();
            Some(removed)
        } else {
            None
        }
    }

    /// Check if batch is full according to config
    pub fn is_full(&self, config: &BatchConfig) -> bool {
        self.jobs.len() >= config.max_jobs_per_batch ||
        self.metadata.estimated_duration >= config.max_batch_duration
    }

    /// Get total resource requirements for the batch
    pub fn total_resource_requirements(&self) -> BatchResourceRequirements {
        let mut total_qubits = 0;
        let mut total_classical = 0;
        let mut max_connectivity = 0;

        for job in &self.jobs {
            total_qubits += job.requirements.qubits_needed;
            total_classical += job.circuit.num_classical;
            max_connectivity = max_connectivity.max(job.requirements.connectivity_requirements.len());
        }

        BatchResourceRequirements {
            total_qubits,
            total_classical,
            max_connectivity_requirements: max_connectivity,
            estimated_duration: self.metadata.estimated_duration,
        }
    }

    /// Update batch metadata based on current jobs
    fn update_metadata(&mut self) {
        self.metadata.estimated_duration = self.jobs.iter()
            .map(|j| j.estimated_duration)
            .sum();

        // Update priority to highest priority job
        self.metadata.priority = self.jobs.iter()
            .map(|j| j.priority)
            .max()
            .unwrap_or(0);

        // Determine batch type based on job characteristics
        self.metadata.batch_type = self.infer_batch_type();
    }

    /// Infer batch type from job characteristics
    fn infer_batch_type(&self) -> BatchType {
        if self.jobs.is_empty() {
            return BatchType::Independent;
        }

        // Check if any job has high priority
        let has_high_priority = self.jobs.iter().any(|j| j.priority > 100);
        if has_high_priority {
            return BatchType::HighPriority;
        }

        // Check for resource overlap (potential parallel execution)
        let mut all_qubits = std::collections::HashSet::new();
        let mut has_overlap = false;

        for job in &self.jobs {
            let job_qubits: std::collections::HashSet<_> = (0..job.requirements.qubits_needed).collect();
            if !all_qubits.is_disjoint(&job_qubits) {
                has_overlap = true;
                break;
            }
            all_qubits.extend(job_qubits);
        }

        if has_overlap {
            BatchType::SharedResource
        } else {
            BatchType::Parallel
        }
    }
}

/// Resource requirements for a batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResourceRequirements {
    /// Total qubits needed
    pub total_qubits: usize,
    /// Total classical bits needed
    pub total_classical: usize,
    /// Maximum connectivity requirements
    pub max_connectivity_requirements: usize,
    /// Estimated total duration
    pub estimated_duration: u64,
}

/// Batch scheduler for creating optimal batches
#[derive(Debug, Clone)]
pub struct BatchScheduler<'a> {
    topology: &'a Topology,
}

impl<'a> BatchScheduler<'a> {
    /// Create a new batch scheduler
    pub fn new(topology: &'a Topology) -> Self {
        Self { topology }
    }

    /// Create batches from a list of jobs
    pub fn create_batches(&self, jobs: Vec<Job>, config: BatchConfig) -> Result<Vec<Batch>> {
        match config.strategy {
            BatchingStrategy::FirstComeFirstServed => {
                self.batch_fcfs(jobs, &config)
            }
            BatchingStrategy::ResourceAware => {
                self.batch_resource_aware(jobs, &config)
            }
            BatchingStrategy::PriorityBased => {
                self.batch_priority_based(jobs, &config)
            }
            BatchingStrategy::DeadlineAware => {
                self.batch_deadline_aware(jobs, &config)
            }
            BatchingStrategy::Adaptive => {
                self.batch_adaptive(jobs, &config)
            }
        }
    }

    /// First-come-first-served batching
    fn batch_fcfs(&self, jobs: Vec<Job>, config: &BatchConfig) -> Result<Vec<Batch>> {
        let mut batches = Vec::new();
        let mut current_batch = Batch::new(0);

        for job in jobs {
            if current_batch.is_full(config) {
                batches.push(current_batch);
                current_batch = Batch::new(batches.len());
            }
            current_batch.add_job(job)?;
        }

        if !current_batch.jobs.is_empty() {
            batches.push(current_batch);
        }

        Ok(batches)
    }

    /// Resource-aware batching
    fn batch_resource_aware(&self, mut jobs: Vec<Job>, config: &BatchConfig) -> Result<Vec<Batch>> {
        // Sort jobs by resource requirements
        jobs.sort_by_key(|job| job.requirements.qubits_needed);

        let mut batches = Vec::new();
        let topology_qubits = self.topology.qubit_count();

        while !jobs.is_empty() {
            let mut batch = Batch::new(batches.len());
            let mut used_qubits = 0;
            let mut i = 0;

            while i < jobs.len() && !batch.is_full(config) {
                let job = &jobs[i];
                
                // Check if job fits in current batch
                if used_qubits + job.requirements.qubits_needed <= topology_qubits {
                    let job = jobs.remove(i);
                    used_qubits += job.requirements.qubits_needed;
                    batch.add_job(job)?;
                } else {
                    i += 1;
                }
            }

            if !batch.jobs.is_empty() {
                batches.push(batch);
            } else {
                // If no job fits, take the first one anyway
                if !jobs.is_empty() {
                    let job = jobs.remove(0);
                    batch.add_job(job)?;
                    batches.push(batch);
                }
            }
        }

        Ok(batches)
    }

    /// Priority-based batching
    fn batch_priority_based(&self, mut jobs: Vec<Job>, config: &BatchConfig) -> Result<Vec<Batch>> {
        // Sort jobs by priority (highest first)
        jobs.sort_by(|a, b| b.priority.cmp(&a.priority));

        // Group jobs by priority levels
        let mut priority_groups: HashMap<i32, Vec<Job>> = HashMap::new();
        for job in jobs {
            priority_groups.entry(job.priority).or_default().push(job);
        }

        let mut batches = Vec::new();
        let mut batch_id = 0;

        // Process each priority level
        for (_priority, priority_jobs) in priority_groups {
            let priority_batches = self.batch_resource_aware(priority_jobs, config)?;
            for mut batch in priority_batches {
                batch.id = batch_id;
                batches.push(batch);
                batch_id += 1;
            }
        }

        Ok(batches)
    }

    /// Deadline-aware batching
    fn batch_deadline_aware(&self, mut jobs: Vec<Job>, config: &BatchConfig) -> Result<Vec<Batch>> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_err(|_| QvmError::internal_error("Time error"))?
            .as_micros() as u64;

        // Sort jobs by deadline urgency
        jobs.sort_by(|a, b| {
            match (a.deadline, b.deadline) {
                (Some(deadline_a), Some(deadline_b)) => {
                    let urgency_a = deadline_a.saturating_sub(current_time + a.estimated_duration);
                    let urgency_b = deadline_b.saturating_sub(current_time + b.estimated_duration);
                    urgency_a.cmp(&urgency_b)
                }
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
        });

        self.batch_resource_aware(jobs, config)
    }

    /// Adaptive batching (combines multiple strategies)
    fn batch_adaptive(&self, jobs: Vec<Job>, config: &BatchConfig) -> Result<Vec<Batch>> {
        // Analyze job characteristics
        let has_deadlines = jobs.iter().any(|j| j.deadline.is_some());
        let has_priorities = jobs.iter().any(|j| j.priority != 0);
        let resource_diversity = self.calculate_resource_diversity(&jobs);

        // Choose strategy based on job characteristics
        let strategy = if has_deadlines {
            BatchingStrategy::DeadlineAware
        } else if has_priorities && resource_diversity > 0.5 {
            BatchingStrategy::PriorityBased
        } else if resource_diversity > 0.3 {
            BatchingStrategy::ResourceAware
        } else {
            BatchingStrategy::FirstComeFirstServed
        };

        let adaptive_config = BatchConfig { strategy, ..config.clone() };
        self.create_batches(jobs, adaptive_config)
    }

    /// Calculate diversity in resource requirements
    fn calculate_resource_diversity(&self, jobs: &[Job]) -> f64 {
        if jobs.is_empty() {
            return 0.0;
        }

        let qubit_requirements: Vec<usize> = jobs.iter()
            .map(|j| j.requirements.qubits_needed)
            .collect();

        let min_qubits = *qubit_requirements.iter().min().unwrap_or(&0);
        let max_qubits = *qubit_requirements.iter().max().unwrap_or(&0);

        if max_qubits == 0 {
            0.0
        } else {
            (max_qubits - min_qubits) as f64 / max_qubits as f64
        }
    }

    /// Optimize batches for better resource utilization
    pub fn optimize_batches(&self, batches: Vec<Batch>) -> Result<Vec<Batch>> {
        let mut optimized = batches;

        // Sort batches by resource efficiency
        optimized.sort_by(|a, b| {
            let efficiency_a = self.calculate_batch_efficiency(a);
            let efficiency_b = self.calculate_batch_efficiency(b);
            efficiency_b.partial_cmp(&efficiency_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Try to merge underutilized batches
        let mut i = 0;
        while i < optimized.len() {
            if i + 1 < optimized.len() {
                if self.can_merge_batches(&optimized[i], &optimized[i + 1]) {
                    let mut merged = optimized.remove(i);
                    let other = optimized.remove(i); // Index shifts after first removal
                    
                    for job in other.jobs {
                        merged.add_job(job)?;
                    }
                    
                    optimized.insert(i, merged);
                    continue; // Check this position again
                }
            }
            i += 1;
        }

        Ok(optimized)
    }

    /// Check if two batches can be merged
    fn can_merge_batches(&self, batch1: &Batch, batch2: &Batch) -> bool {
        let combined_qubits = batch1.total_resource_requirements().total_qubits +
                            batch2.total_resource_requirements().total_qubits;
        
        combined_qubits <= self.topology.qubit_count() &&
        batch1.jobs.len() + batch2.jobs.len() <= 20 // Reasonable limit
    }

    /// Calculate batch efficiency
    fn calculate_batch_efficiency(&self, batch: &Batch) -> f64 {
        if batch.jobs.is_empty() {
            return 0.0;
        }

        let requirements = batch.total_resource_requirements();
        let topology_qubits = self.topology.qubit_count();
        
        if topology_qubits == 0 {
            return 0.0;
        }

        requirements.total_qubits as f64 / topology_qubits as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;
    use crate::circuit_ir::CircuitBuilder;

    #[test]
    fn test_batch_creation() {
        let mut batch = Batch::new(0);
        assert_eq!(batch.id, 0);
        assert!(batch.jobs.is_empty());

        let circuit = CircuitBuilder::new("test", 2, 2).h(0).unwrap().build();
        let job = Job::new(0, circuit);
        
        batch.add_job(job).unwrap();
        assert_eq!(batch.jobs.len(), 1);
    }

    #[test]
    fn test_batch_scheduler_fcfs() {
        let topology = TopologyBuilder::grid(3, 3);
        let scheduler = BatchScheduler::new(&topology);

        let circuit1 = CircuitBuilder::new("test1", 2, 2).h(0).unwrap().build();
        let circuit2 = CircuitBuilder::new("test2", 3, 3).h(0).unwrap().build();
        
        let jobs = vec![
            Job::new(0, circuit1),
            Job::new(1, circuit2),
        ];

        let config = BatchConfig {
            max_jobs_per_batch: 1,
            ..Default::default()
        };

        let batches = scheduler.create_batches(jobs, config).unwrap();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_resource_aware_batching() {
        let topology = TopologyBuilder::grid(4, 4); // 16 qubits
        let scheduler = BatchScheduler::new(&topology);

        let jobs = vec![
            Job::new(0, CircuitBuilder::new("small", 2, 2).h(0).unwrap().build()),
            Job::new(1, CircuitBuilder::new("medium", 5, 5).h(0).unwrap().build()),
            Job::new(2, CircuitBuilder::new("large", 10, 10).h(0).unwrap().build()),
        ];

        let config = BatchConfig {
            strategy: BatchingStrategy::ResourceAware,
            max_jobs_per_batch: 10,
            ..Default::default()
        };

        let batches = scheduler.create_batches(jobs, config).unwrap();
        
        // Should be able to fit smaller jobs together
        let total_jobs: usize = batches.iter().map(|b| b.jobs.len()).sum();
        assert_eq!(total_jobs, 3);
    }

    #[test]
    fn test_batch_optimization() {
        let topology = TopologyBuilder::grid(4, 4);
        let scheduler = BatchScheduler::new(&topology);

        let batches = vec![
            {
                let mut batch = Batch::new(0);
                let circuit = CircuitBuilder::new("small1", 2, 2).h(0).unwrap().build();
                batch.add_job(Job::new(0, circuit)).unwrap();
                batch
            },
            {
                let mut batch = Batch::new(1);
                let circuit = CircuitBuilder::new("small2", 2, 2).h(0).unwrap().build();
                batch.add_job(Job::new(1, circuit)).unwrap();
                batch
            }
        ];

        let optimized = scheduler.optimize_batches(batches).unwrap();
        
        // Should potentially merge small batches
        assert!(!optimized.is_empty());
    }
}