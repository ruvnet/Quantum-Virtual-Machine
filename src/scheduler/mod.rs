//! Quantum circuit scheduling engine

pub mod binpack;
pub mod assignment;
pub mod batch;
pub mod multiplexing;
pub mod optimization;

pub use binpack::*;
pub use assignment::*;
pub use batch::*;
pub use multiplexing::*;
pub use optimization::*;

use crate::{QvmError, Result, QuantumCircuit, Topology};
use crate::topology::{Tile, TileManager, BufferManager, BufferConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Job representing a circuit to be scheduled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    /// Unique job identifier
    pub id: usize,
    /// Quantum circuit to execute
    pub circuit: QuantumCircuit,
    /// Job priority (higher = more priority)
    pub priority: i32,
    /// Earliest start time (timestamp)
    pub earliest_start: Option<u64>,
    /// Latest completion time (timestamp)
    pub deadline: Option<u64>,
    /// Estimated execution time in microseconds
    pub estimated_duration: u64,
    /// Resource requirements
    pub requirements: JobRequirements,
}

/// Job resource requirements
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JobRequirements {
    /// Required number of qubits
    pub qubits_needed: usize,
    /// Required qubit connectivity
    pub connectivity_requirements: Vec<(usize, usize)>,
    /// Minimum tile size needed
    pub min_tile_size: Option<(u32, u32)>,
    /// Buffer zone requirements
    pub buffer_requirements: HashMap<String, usize>,
    /// Special hardware requirements
    pub hardware_requirements: Vec<String>,
}

/// Complete scheduling result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    /// Scheduled assignments
    pub assignments: Vec<Assignment>,
    /// Scheduling metadata
    pub metadata: ScheduleMetadata,
    /// Total schedule duration
    pub total_duration: u64,
    /// Resource utilization statistics
    pub utilization: ResourceUtilization,
}

/// Scheduling metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScheduleMetadata {
    /// Scheduling algorithm used
    pub algorithm: String,
    /// Schedule generation timestamp
    pub generated_at: u64,
    /// Number of jobs scheduled
    pub job_count: usize,
    /// Scheduling success rate
    pub success_rate: f64,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

/// Resource utilization statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// Average qubit utilization (0.0 to 1.0)
    pub avg_qubit_utilization: f64,
    /// Peak qubit utilization
    pub peak_qubit_utilization: f64,
    /// Total idle time
    pub total_idle_time: u64,
    /// Scheduling efficiency
    pub efficiency: f64,
}

impl Job {
    /// Create a new job
    pub fn new(id: usize, circuit: QuantumCircuit) -> Self {
        let estimated_duration = Self::estimate_duration(&circuit);
        let requirements = JobRequirements {
            qubits_needed: circuit.num_qubits,
            connectivity_requirements: circuit.two_qubit_interactions()
                .into_iter()
                .map(|(q1, q2)| (q1.index(), q2.index()))
                .collect(),
            ..Default::default()
        };

        Self {
            id,
            circuit,
            priority: 0,
            earliest_start: None,
            deadline: None,
            estimated_duration,
            requirements,
        }
    }

    /// Create a job with priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set earliest start time
    pub fn with_earliest_start(mut self, timestamp: u64) -> Self {
        self.earliest_start = Some(timestamp);
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, timestamp: u64) -> Self {
        self.deadline = Some(timestamp);
        self
    }

    /// Estimate execution duration based on circuit operations
    fn estimate_duration(circuit: &QuantumCircuit) -> u64 {
        let mut duration = 0u64;
        
        for operation in &circuit.operations {
            duration += (operation.estimated_time() * 1_000_000.0) as u64; // Convert to microseconds
        }
        
        // Add overhead for circuit setup/teardown
        duration + 1000 // 1ms overhead
    }

    /// Check if job can start at given time
    pub fn can_start_at(&self, timestamp: u64) -> bool {
        if let Some(earliest) = self.earliest_start {
            timestamp >= earliest
        } else {
            true
        }
    }

    /// Check if job meets deadline when started at given time
    pub fn meets_deadline(&self, start_time: u64) -> bool {
        if let Some(deadline) = self.deadline {
            start_time + self.estimated_duration <= deadline
        } else {
            true
        }
    }

    /// Get job urgency score (higher = more urgent)
    pub fn urgency_score(&self, current_time: u64) -> f64 {
        let mut score = self.priority as f64;
        
        // Add urgency based on deadline
        if let Some(deadline) = self.deadline {
            let time_to_deadline = deadline.saturating_sub(current_time);
            if time_to_deadline > 0 {
                score += 1000.0 / time_to_deadline as f64;
            } else {
                score += 10000.0; // Overdue
            }
        }
        
        score
    }
}

/// Main scheduler implementation
#[derive(Debug, Clone)]
pub struct Scheduler {
    topology: Topology,
    tile_manager: TileManager,
    buffer_manager: BufferManager,
    bin_packer: BinPacker,
    config: SchedulerConfig,
}

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Scheduling strategy
    pub strategy: SchedulingStrategy,
    /// Maximum scheduling time in milliseconds
    pub max_scheduling_time: u64,
    /// Enable buffer zone enforcement
    pub enforce_buffers: bool,
    /// Buffer configuration
    pub buffer_config: BufferConfig,
    /// Enable load balancing
    pub enable_load_balancing: bool,
    /// Optimization level (0-3)
    pub optimization_level: u8,
}

/// Available scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchedulingStrategy {
    /// First-fit decreasing by size
    FirstFitDecreasing,
    /// Best-fit by resource requirements
    BestFit,
    /// Worst-fit for load balancing
    WorstFit,
    /// Priority-based scheduling
    Priority,
    /// Deadline-aware scheduling
    DeadlineAware,
    /// Hybrid approach
    Adaptive,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            strategy: SchedulingStrategy::Adaptive,
            max_scheduling_time: 1000, // 1 second
            enforce_buffers: true,
            buffer_config: BufferConfig::default(),
            enable_load_balancing: true,
            optimization_level: 2,
        }
    }
}

impl Scheduler {
    /// Create a new scheduler
    pub fn new(topology: Topology) -> Self {
        let buffer_manager = BufferManager::new(BufferConfig::default());
        let bin_packer = BinPacker::new();
        
        Self {
            topology,
            tile_manager: TileManager::new(),
            buffer_manager,
            bin_packer,
            config: SchedulerConfig::default(),
        }
    }

    /// Create scheduler with custom configuration
    pub fn with_config(topology: Topology, config: SchedulerConfig) -> Self {
        let buffer_manager = BufferManager::new(config.buffer_config.clone());
        let bin_packer = BinPacker::new();
        
        Self {
            topology,
            tile_manager: TileManager::new(),
            buffer_manager,
            bin_packer,
            config,
        }
    }

    /// Schedule a collection of jobs
    pub async fn schedule(&mut self, jobs: Vec<Job>) -> Result<Schedule> {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| QvmError::internal_error("Time error"))?
            .as_millis() as u64;

        let jobs_count = jobs.len();
        // Sort jobs by priority and urgency
        let mut sorted_jobs = jobs;
        sorted_jobs.sort_by(|a, b| {
            let urgency_a = a.urgency_score(start_time);
            let urgency_b = b.urgency_score(start_time);
            urgency_b.partial_cmp(&urgency_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply scheduling strategy
        let assignments = match self.config.strategy {
            SchedulingStrategy::FirstFitDecreasing => {
                self.schedule_first_fit_decreasing(sorted_jobs).await?
            }
            SchedulingStrategy::BestFit => {
                self.schedule_best_fit(sorted_jobs).await?
            }
            SchedulingStrategy::Priority => {
                self.schedule_priority(sorted_jobs).await?
            }
            SchedulingStrategy::DeadlineAware => {
                self.schedule_deadline_aware(sorted_jobs).await?
            }
            SchedulingStrategy::Adaptive => {
                self.schedule_adaptive(sorted_jobs).await?
            }
            _ => {
                self.schedule_first_fit_decreasing(sorted_jobs).await?
            }
        };

        let total_duration = assignments.iter()
            .map(|a| a.start_time + a.duration)
            .max()
            .unwrap_or(0);

        let utilization = self.calculate_utilization(&assignments, total_duration);

        let metadata = ScheduleMetadata {
            algorithm: format!("{:?}", self.config.strategy),
            generated_at: start_time,
            job_count: assignments.len(),
            success_rate: assignments.len() as f64 / jobs_count as f64,
            ..Default::default()
        };

        Ok(Schedule {
            assignments,
            metadata,
            total_duration,
            utilization,
        })
    }

    /// First-fit decreasing scheduling
    async fn schedule_first_fit_decreasing(&mut self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        let mut assignments = Vec::new();
        let batch_scheduler = BatchScheduler::new(&self.topology);

        // Group jobs into batches
        let batches = batch_scheduler.create_batches(jobs, BatchConfig::default())?;

        for batch in batches {
            let batch_assignments = self.bin_packer.pack_batch(batch, &self.topology).await?;
            assignments.extend(batch_assignments);
        }

        Ok(assignments)
    }

    /// Best-fit scheduling
    async fn schedule_best_fit(&mut self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        // Simplified best-fit implementation
        self.schedule_first_fit_decreasing(jobs).await
    }

    /// Priority-based scheduling
    async fn schedule_priority(&mut self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        // Jobs are already sorted by priority/urgency
        self.schedule_first_fit_decreasing(jobs).await
    }

    /// Deadline-aware scheduling
    async fn schedule_deadline_aware(&mut self, mut jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        // Sort by deadline urgency
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| QvmError::internal_error("Time error"))?
            .as_micros() as u64;

        jobs.sort_by(|a, b| {
            match (a.deadline, b.deadline) {
                (Some(deadline_a), Some(deadline_b)) => deadline_a.cmp(&deadline_b),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.priority.cmp(&b.priority).reverse(),
            }
        });

        self.schedule_first_fit_decreasing(jobs).await
    }

    /// Adaptive scheduling (combines multiple strategies)
    async fn schedule_adaptive(&mut self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        // Use deadline-aware for jobs with deadlines, priority for others
        let (deadline_jobs, other_jobs): (Vec<_>, Vec<_>) = jobs.into_iter()
            .partition(|job| job.deadline.is_some());

        let mut assignments = Vec::new();

        // Schedule deadline jobs first
        if !deadline_jobs.is_empty() {
            let deadline_assignments = self.schedule_deadline_aware(deadline_jobs).await?;
            assignments.extend(deadline_assignments);
        }

        // Schedule remaining jobs
        if !other_jobs.is_empty() {
            let other_assignments = self.schedule_priority(other_jobs).await?;
            assignments.extend(other_assignments);
        }

        Ok(assignments)
    }

    /// Calculate resource utilization statistics
    fn calculate_utilization(&self, assignments: &[Assignment], total_duration: u64) -> ResourceUtilization {
        if assignments.is_empty() || total_duration == 0 {
            return ResourceUtilization::default();
        }

        let total_qubits = self.topology.qubit_count();
        let mut qubit_usage_time = vec![0u64; total_qubits];

        // Calculate per-qubit usage
        for assignment in assignments {
            for &qubit_idx in &assignment.qubit_mapping {
                if qubit_idx < total_qubits {
                    qubit_usage_time[qubit_idx] += assignment.duration;
                }
            }
        }

        let total_possible_time = total_qubits as u64 * total_duration;
        let total_used_time: u64 = qubit_usage_time.iter().sum();

        let avg_qubit_utilization = if total_possible_time > 0 {
            total_used_time as f64 / total_possible_time as f64
        } else {
            0.0
        };

        let peak_utilization = qubit_usage_time.iter()
            .map(|&usage| usage as f64 / total_duration as f64)
            .fold(0.0, f64::max);

        let efficiency = if !assignments.is_empty() {
            let ideal_time: u64 = assignments.iter().map(|a| a.duration).sum();
            ideal_time as f64 / total_duration as f64
        } else {
            0.0
        };

        ResourceUtilization {
            avg_qubit_utilization,
            peak_qubit_utilization: peak_utilization,
            total_idle_time: total_possible_time.saturating_sub(total_used_time),
            efficiency,
        }
    }

    /// Get scheduler statistics
    pub fn statistics(&self) -> SchedulerStatistics {
        SchedulerStatistics {
            total_qubits: self.topology.qubit_count(),
            available_tiles: self.tile_manager.available_tiles().len(),
            active_buffers: 0, // Would need to track this
            avg_scheduling_time: 0.0, // Would need to track this
        }
    }
}

/// Scheduler statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStatistics {
    pub total_qubits: usize,
    pub available_tiles: usize,
    pub active_buffers: usize,
    pub avg_scheduling_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;
    use crate::circuit_ir::CircuitBuilder;

    #[test]
    fn test_job_creation() {
        let circuit = CircuitBuilder::new("test", 2, 2)
            .h(0).unwrap()
            .cx(0, 1).unwrap()
            .build();

        let job = Job::new(0, circuit);
        assert_eq!(job.id, 0);
        assert_eq!(job.requirements.qubits_needed, 2);
        assert!(job.estimated_duration > 0);
    }

    #[test]
    fn test_scheduler_creation() {
        let topology = TopologyBuilder::grid(3, 3);
        let scheduler = Scheduler::new(topology);
        assert_eq!(scheduler.topology.qubit_count(), 9);
    }

    #[tokio::test]
    async fn test_empty_schedule() {
        let topology = TopologyBuilder::grid(2, 2);
        let mut scheduler = Scheduler::new(topology);
        
        let schedule = scheduler.schedule(vec![]).await.unwrap();
        assert!(schedule.assignments.is_empty());
        assert_eq!(schedule.total_duration, 0);
    }

    #[tokio::test]
    async fn test_single_job_schedule() {
        let topology = TopologyBuilder::grid(2, 2);
        let mut scheduler = Scheduler::new(topology);
        
        let circuit = CircuitBuilder::new("test", 2, 2)
            .h(0).unwrap()
            .cx(0, 1).unwrap()
            .build();
        let job = Job::new(0, circuit);

        let schedule = scheduler.schedule(vec![job]).await.unwrap();
        assert_eq!(schedule.assignments.len(), 1);
        assert!(schedule.total_duration > 0);
        assert_eq!(schedule.metadata.job_count, 1);
    }
}