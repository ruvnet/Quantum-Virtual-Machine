//! Spatial and temporal multiplexing for quantum circuit scheduling

use crate::{QvmError, Result, Topology};
use crate::scheduler::{Job, Assignment, BinPacker, BinPackingAlgorithm};
use crate::topology::{TileFinder, TilePreferences, TileManager};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Spatial multiplexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialMultiplexConfig {
    /// Enable spatial multiplexing
    pub enabled: bool,
    /// Maximum number of parallel jobs
    pub max_parallel_jobs: usize,
    /// Minimum distance between parallel jobs
    pub min_job_separation: usize,
    /// Buffer zone size for crosstalk mitigation
    pub buffer_zone_size: usize,
    /// Spatial isolation strategy
    pub isolation_strategy: SpatialIsolationStrategy,
}

impl Default for SpatialMultiplexConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_parallel_jobs: 4,
            min_job_separation: 2,
            buffer_zone_size: 1,
            isolation_strategy: SpatialIsolationStrategy::TileBased,
        }
    }
}

/// Spatial isolation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialIsolationStrategy {
    /// Tile-based isolation using non-overlapping tiles
    TileBased,
    /// Distance-based isolation using minimum qubit distances
    DistanceBased,
    /// Buffer zone isolation with dedicated buffer qubits
    BufferZoned,
    /// Connectivity-based isolation using graph partitioning
    ConnectivityBased,
}

/// Temporal multiplexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMultiplexConfig {
    /// Enable temporal multiplexing
    pub enabled: bool,
    /// Minimum batch size for temporal scheduling
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Time slice duration (microseconds)
    pub time_slice_duration: u64,
    /// Overlap tolerance between batches
    pub overlap_tolerance: u64,
    /// Temporal scheduling strategy
    pub scheduling_strategy: TemporalSchedulingStrategy,
}

impl Default for TemporalMultiplexConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_batch_size: 2,
            max_batch_size: 10,
            time_slice_duration: 100_000, // 100ms
            overlap_tolerance: 5_000,     // 5ms
            scheduling_strategy: TemporalSchedulingStrategy::RoundRobin,
        }
    }
}

/// Temporal scheduling strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalSchedulingStrategy {
    /// Round-robin scheduling across time slices
    RoundRobin,
    /// Priority-based temporal scheduling
    PriorityBased,
    /// Deadline-aware temporal scheduling
    DeadlineAware,
    /// Load-balancing temporal scheduling
    LoadBalanced,
}

/// Spatial multiplexer for parallel job execution
#[derive(Debug, Clone)]
pub struct SpatialMultiplexer {
    topology: Topology,
    tile_manager: TileManager,
    config: SpatialMultiplexConfig,
}

impl SpatialMultiplexer {
    /// Create a new spatial multiplexer
    pub fn new(topology: Topology, config: SpatialMultiplexConfig) -> Self {
        let tile_manager = TileManager::new();
        
        Self {
            topology,
            tile_manager,
            config,
        }
    }

    /// Schedule jobs using spatial multiplexing
    pub async fn schedule_spatially(&mut self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        if !self.config.enabled {
            // Fall back to sequential scheduling
            return self.schedule_sequentially(jobs).await;
        }

        // Group jobs that can be executed in parallel
        let parallel_groups = self.create_spatial_groups(jobs)?;
        let mut all_assignments = Vec::new();

        for group in parallel_groups {
            let assignments = self.schedule_parallel_group(group).await?;
            all_assignments.extend(assignments);
        }

        Ok(all_assignments)
    }

    /// Create groups of jobs that can be executed spatially in parallel
    fn create_spatial_groups(&self, jobs: Vec<Job>) -> Result<Vec<Vec<Job>>> {
        let mut groups = Vec::new();
        let mut remaining_jobs = jobs;

        while !remaining_jobs.is_empty() {
            let mut current_group = Vec::new();
            let mut used_tiles = HashSet::new();
            let mut i = 0;

            // Greedy selection for current parallel group
            while i < remaining_jobs.len() && current_group.len() < self.config.max_parallel_jobs {
                let job = &remaining_jobs[i];
                
                if let Ok(tile) = self.find_suitable_tile_for_job(job, &used_tiles) {
                    current_group.push(remaining_jobs.remove(i));
                    used_tiles.insert(tile.id);
                } else {
                    i += 1;
                }
            }

            // If no jobs could be added to current group, take the first remaining job
            if current_group.is_empty() && !remaining_jobs.is_empty() {
                current_group.push(remaining_jobs.remove(0));
            }

            if !current_group.is_empty() {
                groups.push(current_group);
            }
        }

        Ok(groups)
    }

    /// Schedule a group of jobs to execute in parallel
    async fn schedule_parallel_group(&mut self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        match self.config.isolation_strategy {
            SpatialIsolationStrategy::TileBased => {
                self.schedule_tile_based(jobs).await
            }
            SpatialIsolationStrategy::DistanceBased => {
                self.schedule_distance_based(jobs).await
            }
            SpatialIsolationStrategy::BufferZoned => {
                self.schedule_buffer_zoned(jobs).await
            }
            SpatialIsolationStrategy::ConnectivityBased => {
                self.schedule_connectivity_based(jobs).await
            }
        }
    }

    /// Tile-based parallel scheduling
    async fn schedule_tile_based(&mut self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        let mut assignments = Vec::new();
        let start_time = 0u64; // All jobs in group start simultaneously

        for job in jobs {
            let preferences = TilePreferences {
                min_width: 2,
                min_height: 2,
                max_qubits: job.requirements.qubits_needed * 2,
                buffer_size: self.config.buffer_zone_size,
                prefer_center: false, // Prefer edges for better isolation
                min_connectivity: 0.5,
                ..Default::default()
            };

            let tile_finder = TileFinder::new(&self.topology);
            let tile = tile_finder
                .find_best_tile(job.requirements.qubits_needed, &preferences)?
                .ok_or_else(|| QvmError::scheduling_error("No suitable tile found for spatial multiplexing"))?;

            let qubit_mapping: Vec<usize> = tile.qubits
                .iter()
                .take(job.requirements.qubits_needed)
                .map(|q| q.index())
                .collect();

            let assignment = Assignment {
                job_id: job.id,
                tile_id: tile.id,
                start_time,
                duration: job.estimated_duration,
                qubit_mapping,
                classical_mapping: (0..job.circuit.num_classical).collect(),
                resource_allocation: Default::default(),
            };

            assignments.push(assignment);
        }

        Ok(assignments)
    }

    /// Distance-based parallel scheduling
    async fn schedule_distance_based(&mut self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        let mut assignments = Vec::new();
        let mut used_qubits = HashSet::new();
        let start_time = 0u64;

        for job in jobs {
            // Find qubits that are far enough from already used qubits
            let suitable_qubits = self.find_distant_qubits(&used_qubits, job.requirements.qubits_needed)?;
            
            if suitable_qubits.len() < job.requirements.qubits_needed {
                return Err(QvmError::scheduling_error("Not enough distant qubits available"));
            }

            let qubit_mapping: Vec<usize> = suitable_qubits
                .into_iter()
                .take(job.requirements.qubits_needed)
                .collect();

            // Mark qubits as used (including buffer zone)
            for &qubit_idx in &qubit_mapping {
                let buffer_qubits = self.topology.qubits_within_distance(
                    qubit_idx.into(), 
                    self.config.min_job_separation as u32
                );
                for (buffer_qubit, _) in buffer_qubits {
                    used_qubits.insert(buffer_qubit.index());
                }
            }

            let assignment = Assignment {
                job_id: job.id,
                tile_id: 0, // No specific tile for distance-based
                start_time,
                duration: job.estimated_duration,
                qubit_mapping,
                classical_mapping: (0..job.circuit.num_classical).collect(),
                resource_allocation: Default::default(),
            };

            assignments.push(assignment);
        }

        Ok(assignments)
    }

    /// Buffer-zoned parallel scheduling
    async fn schedule_buffer_zoned(&mut self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        // Implement buffer zone scheduling with explicit buffer qubit allocation
        let mut assignments = Vec::new();
        let mut used_qubits = HashSet::new();
        let start_time = 0u64;

        for job in jobs {
            // Find qubits with sufficient buffer zones
            let (job_qubits, buffer_qubits) = self.find_qubits_with_buffers(
                &used_qubits, 
                job.requirements.qubits_needed
            )?;

            let assignment = Assignment {
                job_id: job.id,
                tile_id: 0,
                start_time,
                duration: job.estimated_duration,
                qubit_mapping: job_qubits.clone(),
                classical_mapping: (0..job.circuit.num_classical).collect(),
                resource_allocation: crate::scheduler::ResourceAllocation {
                    buffer_qubits,
                    ..Default::default()
                },
            };

            // Mark all qubits (job + buffer) as used
            for &qubit in &assignment.qubit_mapping {
                used_qubits.insert(qubit);
            }
            for &buffer in &assignment.resource_allocation.buffer_qubits {
                used_qubits.insert(buffer);
            }

            assignments.push(assignment);
        }

        Ok(assignments)
    }

    /// Connectivity-based parallel scheduling
    async fn schedule_connectivity_based(&mut self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        // Implement graph partitioning-based scheduling
        let partitions = self.partition_topology(jobs.len())?;
        let mut assignments = Vec::new();
        let start_time = 0u64;

        for (job, partition) in jobs.into_iter().zip(partitions) {
            if partition.len() < job.requirements.qubits_needed {
                return Err(QvmError::scheduling_error("Partition too small for job requirements"));
            }

            let qubit_mapping: Vec<usize> = partition
                .into_iter()
                .take(job.requirements.qubits_needed)
                .collect();

            let assignment = Assignment {
                job_id: job.id,
                tile_id: 0,
                start_time,
                duration: job.estimated_duration,
                qubit_mapping,
                classical_mapping: (0..job.circuit.num_classical).collect(),
                resource_allocation: Default::default(),
            };

            assignments.push(assignment);
        }

        Ok(assignments)
    }

    /// Fall back to sequential scheduling
    async fn schedule_sequentially(&self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        let bin_packer = BinPacker::with_algorithm(BinPackingAlgorithm::FirstFitDecreasing);
        let batch = crate::scheduler::batch::Batch {
            id: 0,
            jobs,
            metadata: Default::default(),
        };
        bin_packer.pack_batch(batch, &self.topology).await
    }

    /// Helper methods
    fn find_suitable_tile_for_job(&self, job: &Job, used_tiles: &HashSet<usize>) -> Result<crate::topology::Tile> {
        // Simplified tile finding - in practice, you'd need proper implementation
        Err(QvmError::scheduling_error("Tile finding not implemented"))
    }

    fn find_distant_qubits(&self, used_qubits: &HashSet<usize>, needed: usize) -> Result<Vec<usize>> {
        let mut suitable = Vec::new();
        
        for qubit in self.topology.qubits() {
            let qubit_idx = qubit.index();
            if used_qubits.contains(&qubit_idx) {
                continue;
            }

            // Check if this qubit is far enough from all used qubits
            let is_distant = used_qubits.iter().all(|&used_idx| {
                if let Some(path) = self.topology.shortest_path(qubit, used_idx.into()) {
                    path.len() > self.config.min_job_separation
                } else {
                    true // No path means they're isolated
                }
            });

            if is_distant {
                suitable.push(qubit_idx);
                if suitable.len() >= needed {
                    break;
                }
            }
        }

        Ok(suitable)
    }

    fn find_qubits_with_buffers(&self, used_qubits: &HashSet<usize>, needed: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        let mut job_qubits = Vec::new();
        let mut buffer_qubits = Vec::new();
        
        for qubit in self.topology.qubits() {
            let qubit_idx = qubit.index();
            if used_qubits.contains(&qubit_idx) {
                continue;
            }

            // Find buffer zone for this qubit
            let buffer_zone = self.topology.qubits_within_distance(
                qubit, 
                self.config.buffer_zone_size as u32
            );
            
            // Check if buffer zone is available
            let buffer_available = buffer_zone.iter().all(|(buffer_qubit, _)| {
                !used_qubits.contains(&buffer_qubit.index())
            });

            if buffer_available {
                job_qubits.push(qubit_idx);
                buffer_qubits.extend(
                    buffer_zone.into_iter()
                        .filter(|(q, _)| q.index() != qubit_idx)
                        .map(|(q, _)| q.index())
                );
                
                if job_qubits.len() >= needed {
                    break;
                }
            }
        }

        Ok((job_qubits, buffer_qubits))
    }

    fn partition_topology(&self, num_partitions: usize) -> Result<Vec<Vec<usize>>> {
        // Simplified graph partitioning - in practice, you'd use sophisticated algorithms
        let qubits: Vec<_> = self.topology.qubits().into_iter().map(|q| q.index()).collect();
        let partition_size = qubits.len() / num_partitions;
        
        let mut partitions = Vec::new();
        for i in 0..num_partitions {
            let start = i * partition_size;
            let end = if i == num_partitions - 1 { qubits.len() } else { (i + 1) * partition_size };
            partitions.push(qubits[start..end].to_vec());
        }
        
        Ok(partitions)
    }
}

/// Temporal multiplexer for sequential batch execution
#[derive(Debug, Clone)]
pub struct TemporalMultiplexer {
    config: TemporalMultiplexConfig,
}

impl TemporalMultiplexer {
    /// Create a new temporal multiplexer
    pub fn new(config: TemporalMultiplexConfig) -> Self {
        Self { config }
    }

    /// Schedule jobs using temporal multiplexing
    pub async fn schedule_temporally(&self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        if !self.config.enabled {
            return Ok(vec![]); // Delegate to main scheduler
        }

        // Create temporal batches
        let batches = self.create_temporal_batches(jobs)?;
        let mut all_assignments = Vec::new();
        let mut current_time = 0u64;

        for batch in batches {
            let batch_assignments = self.schedule_temporal_batch(batch, current_time).await?;
            
            // Update current time for next batch
            if let Some(max_end_time) = batch_assignments.iter()
                .map(|a| a.start_time + a.duration)
                .max() {
                current_time = max_end_time + self.config.overlap_tolerance;
            }

            all_assignments.extend(batch_assignments);
        }

        Ok(all_assignments)
    }

    /// Create temporal batches based on configuration
    fn create_temporal_batches(&self, mut jobs: Vec<Job>) -> Result<Vec<Vec<Job>>> {
        match self.config.scheduling_strategy {
            TemporalSchedulingStrategy::RoundRobin => {
                self.create_round_robin_batches(jobs)
            }
            TemporalSchedulingStrategy::PriorityBased => {
                jobs.sort_by_key(|job| std::cmp::Reverse(job.priority));
                self.create_fixed_size_batches(jobs)
            }
            TemporalSchedulingStrategy::DeadlineAware => {
                jobs.sort_by_key(|job| job.deadline.unwrap_or(u64::MAX));
                self.create_fixed_size_batches(jobs)
            }
            TemporalSchedulingStrategy::LoadBalanced => {
                self.create_load_balanced_batches(jobs)
            }
        }
    }

    fn create_round_robin_batches(&self, jobs: Vec<Job>) -> Result<Vec<Vec<Job>>> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();

        for job in jobs {
            current_batch.push(job);
            
            if current_batch.len() >= self.config.max_batch_size {
                batches.push(current_batch);
                current_batch = Vec::new();
            }
        }

        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        Ok(batches)
    }

    fn create_fixed_size_batches(&self, jobs: Vec<Job>) -> Result<Vec<Vec<Job>>> {
        let mut batches = Vec::new();
        let chunk_size = self.config.max_batch_size;

        for chunk in jobs.chunks(chunk_size) {
            batches.push(chunk.to_vec());
        }

        Ok(batches)
    }

    fn create_load_balanced_batches(&self, jobs: Vec<Job>) -> Result<Vec<Vec<Job>>> {
        // Balance batches by total execution time
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_batch_duration = 0u64;

        for job in jobs {
            if current_batch_duration + job.estimated_duration > self.config.time_slice_duration 
                && current_batch.len() >= self.config.min_batch_size {
                batches.push(current_batch);
                current_batch = Vec::new();
                current_batch_duration = 0;
            }

            current_batch_duration += job.estimated_duration;
            current_batch.push(job);

            if current_batch.len() >= self.config.max_batch_size {
                batches.push(current_batch);
                current_batch = Vec::new();
                current_batch_duration = 0;
            }
        }

        if !current_batch.is_empty() {
            batches.push(current_batch);
        }

        Ok(batches)
    }

    async fn schedule_temporal_batch(&self, jobs: Vec<Job>, start_time: u64) -> Result<Vec<Assignment>> {
        // Create assignments for a temporal batch
        // In practice, this would delegate to the main scheduler
        let mut assignments = Vec::new();

        for (i, job) in jobs.into_iter().enumerate() {
            let assignment = Assignment {
                job_id: job.id,
                tile_id: 0,
                start_time: start_time + (i as u64 * 1000), // Simple time offset
                duration: job.estimated_duration,
                qubit_mapping: (0..job.requirements.qubits_needed).collect(),
                classical_mapping: (0..job.circuit.num_classical).collect(),
                resource_allocation: Default::default(),
            };
            assignments.push(assignment);
        }

        Ok(assignments)
    }
}

/// Combined spatial and temporal multiplexer
#[derive(Debug, Clone)]
pub struct HybridMultiplexer {
    spatial: SpatialMultiplexer,
    temporal: TemporalMultiplexer,
}

impl HybridMultiplexer {
    /// Create a new hybrid multiplexer
    pub fn new(
        topology: Topology,
        spatial_config: SpatialMultiplexConfig,
        temporal_config: TemporalMultiplexConfig,
    ) -> Self {
        Self {
            spatial: SpatialMultiplexer::new(topology, spatial_config),
            temporal: TemporalMultiplexer::new(temporal_config),
        }
    }

    /// Schedule jobs using both spatial and temporal multiplexing
    pub async fn schedule_hybrid(&mut self, jobs: Vec<Job>) -> Result<Vec<Assignment>> {
        // First apply temporal multiplexing to create time-ordered batches
        let temporal_batches = self.temporal.create_temporal_batches(jobs)?;
        let mut all_assignments = Vec::new();
        let mut current_time = 0u64;

        for batch in temporal_batches {
            // Then apply spatial multiplexing within each temporal batch
            let mut spatial_assignments = self.spatial.schedule_spatially(batch).await?;
            
            // Adjust timing for temporal ordering
            for assignment in &mut spatial_assignments {
                assignment.start_time += current_time;
            }

            // Update current time for next batch
            if let Some(max_end_time) = spatial_assignments.iter()
                .map(|a| a.start_time + a.duration)
                .max() {
                current_time = max_end_time + self.temporal.config.overlap_tolerance;
            }

            all_assignments.extend(spatial_assignments);
        }

        Ok(all_assignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;
    use crate::circuit_ir::CircuitBuilder;

    #[test]
    fn test_spatial_multiplex_config() {
        let config = SpatialMultiplexConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_parallel_jobs, 4);
        assert_eq!(config.isolation_strategy, SpatialIsolationStrategy::TileBased);
    }

    #[test]
    fn test_temporal_multiplex_config() {
        let config = TemporalMultiplexConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_batch_size, 2);
        assert_eq!(config.scheduling_strategy, TemporalSchedulingStrategy::RoundRobin);
    }

    #[tokio::test]
    async fn test_temporal_multiplexer() {
        let config = TemporalMultiplexConfig::default();
        let multiplexer = TemporalMultiplexer::new(config);

        let circuit = CircuitBuilder::new("test", 2, 2).h(0).unwrap().build();
        let jobs = vec![
            Job::new(0, circuit.clone()),
            Job::new(1, circuit.clone()),
            Job::new(2, circuit),
        ];

        let assignments = multiplexer.schedule_temporally(jobs).await.unwrap();
        assert_eq!(assignments.len(), 3);
        
        // Check that assignments are time-ordered
        for i in 1..assignments.len() {
            assert!(assignments[i].start_time >= assignments[i-1].start_time);
        }
    }
}