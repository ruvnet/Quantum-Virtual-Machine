//! Bin packing algorithms for circuit scheduling

use crate::{QvmError, Result, Topology};
use crate::scheduler::{Job, Assignment};
use crate::scheduler::batch::{Batch, BatchConfig};
use crate::topology::{TileFinder, TilePreferences};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Bin packing algorithm implementations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinPackingAlgorithm {
    /// First fit decreasing
    FirstFitDecreasing,
    /// Best fit decreasing
    BestFitDecreasing,
    /// Worst fit decreasing
    WorstFitDecreasing,
    /// Next fit
    NextFit,
}

/// Bin packer for scheduling quantum circuits
#[derive(Debug, Clone)]
pub struct BinPacker {
    algorithm: BinPackingAlgorithm,
}

impl BinPacker {
    /// Create a new bin packer
    pub fn new() -> Self {
        Self {
            algorithm: BinPackingAlgorithm::FirstFitDecreasing,
        }
    }

    /// Create bin packer with specific algorithm
    pub fn with_algorithm(algorithm: BinPackingAlgorithm) -> Self {
        Self { algorithm }
    }

    /// Pack a batch of jobs onto the topology
    pub async fn pack_batch(&self, batch: Batch, topology: &Topology) -> Result<Vec<Assignment>> {
        match self.algorithm {
            BinPackingAlgorithm::FirstFitDecreasing => {
                self.first_fit_decreasing(batch, topology).await
            }
            BinPackingAlgorithm::BestFitDecreasing => {
                self.best_fit_decreasing(batch, topology).await
            }
            BinPackingAlgorithm::WorstFitDecreasing => {
                self.worst_fit_decreasing(batch, topology).await
            }
            BinPackingAlgorithm::NextFit => {
                self.next_fit(batch, topology).await
            }
        }
    }

    /// First fit decreasing algorithm
    async fn first_fit_decreasing(&self, mut batch: Batch, topology: &Topology) -> Result<Vec<Assignment>> {
        // Sort jobs by size (decreasing)
        batch.jobs.sort_by(|a, b| {
            b.requirements.qubits_needed.cmp(&a.requirements.qubits_needed)
        });

        let mut assignments = Vec::new();
        let tile_finder = TileFinder::new(topology);
        let mut bins: Vec<BinState> = Vec::new();

        for job in batch.jobs {
            // Find the first bin that fits
            let first_fit_idx = self.find_first_fit_bin(&job, &bins)?;
            
            let assignment = if let Some(bin_idx) = first_fit_idx {
                // Use existing bin
                let bin = &mut bins[bin_idx];
                let assignment = self.create_assignment_in_bin(&job, bin, &tile_finder)?;
                bin.used_qubits += job.requirements.qubits_needed;
                bin.end_time = bin.end_time.max(assignment.start_time + assignment.duration);
                assignment
            } else {
                // Create new bin
                let assignment = self.find_assignment_for_job(&job, &tile_finder, 0)?;
                let bin = BinState {
                    id: bins.len(),
                    capacity: topology.qubit_count(),
                    used_qubits: job.requirements.qubits_needed,
                    end_time: assignment.duration,
                };
                bins.push(bin);
                assignment
            };

            assignments.push(assignment);
        }

        Ok(assignments)
    }

    /// Best fit decreasing algorithm
    async fn best_fit_decreasing(&self, mut batch: Batch, topology: &Topology) -> Result<Vec<Assignment>> {
        // Sort jobs by size (decreasing order)
        batch.jobs.sort_by(|a, b| {
            b.requirements.qubits_needed.cmp(&a.requirements.qubits_needed)
        });

        let mut assignments = Vec::new();
        let tile_finder = TileFinder::new(topology);
        let mut bins: Vec<BinState> = Vec::new();

        for job in batch.jobs {
            // Find the best fitting bin (smallest waste)
            let best_bin_idx = self.find_best_fit_bin(&job, &bins)?;
            
            let assignment = if let Some(bin_idx) = best_bin_idx {
                // Use existing bin
                let bin = &mut bins[bin_idx];
                let assignment = self.create_assignment_in_bin(&job, bin, &tile_finder)?;
                bin.used_qubits += job.requirements.qubits_needed;
                bin.end_time = bin.end_time.max(assignment.start_time + assignment.duration);
                assignment
            } else {
                // Create new bin
                let assignment = self.find_assignment_for_job(&job, &tile_finder, 0)?;
                let bin = BinState {
                    capacity: topology.qubit_count(),
                    used_qubits: job.requirements.qubits_needed,
                    end_time: assignment.duration,
                    id: bins.len(),
                };
                bins.push(bin);
                assignment
            };

            assignments.push(assignment);
        }

        Ok(assignments)
    }

    /// Worst fit decreasing algorithm
    async fn worst_fit_decreasing(&self, mut batch: Batch, topology: &Topology) -> Result<Vec<Assignment>> {
        // Sort jobs by size (decreasing order)
        batch.jobs.sort_by(|a, b| {
            b.requirements.qubits_needed.cmp(&a.requirements.qubits_needed)
        });

        let mut assignments = Vec::new();
        let tile_finder = TileFinder::new(topology);
        let mut bins: Vec<BinState> = Vec::new();

        for job in batch.jobs {
            // Find the worst fitting bin (largest waste)
            let worst_bin_idx = self.find_worst_fit_bin(&job, &bins)?;
            
            let assignment = if let Some(bin_idx) = worst_bin_idx {
                // Use existing bin with most space
                let bin = &mut bins[bin_idx];
                let assignment = self.create_assignment_in_bin(&job, bin, &tile_finder)?;
                bin.used_qubits += job.requirements.qubits_needed;
                bin.end_time = bin.end_time.max(assignment.start_time + assignment.duration);
                assignment
            } else {
                // Create new bin
                let assignment = self.find_assignment_for_job(&job, &tile_finder, 0)?;
                let bin = BinState {
                    capacity: topology.qubit_count(),
                    used_qubits: job.requirements.qubits_needed,
                    end_time: assignment.duration,
                    id: bins.len(),
                };
                bins.push(bin);
                assignment
            };

            assignments.push(assignment);
        }

        Ok(assignments)
    }

    /// Next fit decreasing algorithm (NFD)
    async fn next_fit(&self, mut batch: Batch, topology: &Topology) -> Result<Vec<Assignment>> {
        // Sort jobs by size (decreasing) for NFD
        batch.jobs.sort_by(|a, b| {
            b.requirements.qubits_needed.cmp(&a.requirements.qubits_needed)
        });

        let mut assignments = Vec::new();
        let tile_finder = TileFinder::new(topology);
        let mut current_bin = None::<BinState>;
        let mut bin_counter = 0;

        for job in batch.jobs {
            // Check if job fits in current bin
            let fits_current = if let Some(ref bin) = current_bin {
                let remaining = bin.capacity.saturating_sub(bin.used_qubits);
                remaining >= job.requirements.qubits_needed
            } else {
                false
            };

            if fits_current {
                // Use current bin
                if let Some(ref mut bin) = current_bin {
                    let assignment = self.create_assignment_in_bin(&job, bin, &tile_finder)?;
                    bin.used_qubits += job.requirements.qubits_needed;
                    bin.end_time = bin.end_time.max(assignment.start_time + assignment.duration);
                    assignments.push(assignment);
                }
            } else {
                // Create new bin
                let assignment = self.find_assignment_for_job(&job, &tile_finder, 0)?;
                current_bin = Some(BinState {
                    id: bin_counter,
                    capacity: topology.qubit_count(),
                    used_qubits: job.requirements.qubits_needed,
                    end_time: assignment.duration,
                });
                bin_counter += 1;
                assignments.push(assignment);
            }
        }

        Ok(assignments)
    }

    /// Find assignment for a single job
    fn find_assignment_for_job(
        &self,
        job: &Job,
        tile_finder: &TileFinder,
        start_time: u64,
    ) -> Result<Assignment> {
        let preferences = TilePreferences {
            min_width: 2,
            min_height: 2,
            max_qubits: job.requirements.qubits_needed * 2,
            ..Default::default()
        };

        let tile = tile_finder.find_best_tile(job.requirements.qubits_needed, &preferences)?
            .ok_or_else(|| QvmError::scheduling_error("No suitable tile found for job"))?;

        // Create qubit mapping
        let qubit_mapping: Vec<usize> = tile.qubits
            .iter()
            .take(job.requirements.qubits_needed)
            .map(|q| q.index())
            .collect();

        Ok(Assignment {
            job_id: job.id,
            tile_id: tile.id,
            start_time,
            duration: job.estimated_duration,
            qubit_mapping,
            classical_mapping: (0..job.circuit.num_classical).collect(),
            resource_allocation: Default::default(),
        })
    }

    /// Find the first bin that fits the job
    fn find_first_fit_bin(&self, job: &Job, bins: &[BinState]) -> Result<Option<usize>> {
        for (i, bin) in bins.iter().enumerate() {
            let remaining_capacity = bin.capacity.saturating_sub(bin.used_qubits);
            if remaining_capacity >= job.requirements.qubits_needed {
                return Ok(Some(i));
            }
        }
        Ok(None)
    }

    /// Find the best fitting bin for a job (smallest waste)
    fn find_best_fit_bin(&self, job: &Job, bins: &[BinState]) -> Result<Option<usize>> {
        let mut best_fit_idx = None;
        let mut best_fit_waste = usize::MAX;

        for (i, bin) in bins.iter().enumerate() {
            let remaining_capacity = bin.capacity.saturating_sub(bin.used_qubits);
            
            if remaining_capacity >= job.requirements.qubits_needed {
                let waste = remaining_capacity - job.requirements.qubits_needed;
                if waste < best_fit_waste {
                    best_fit_waste = waste;
                    best_fit_idx = Some(i);
                }
            }
        }

        Ok(best_fit_idx)
    }

    /// Find the worst fitting bin for a job (largest waste, for load balancing)
    fn find_worst_fit_bin(&self, job: &Job, bins: &[BinState]) -> Result<Option<usize>> {
        let mut worst_fit_idx = None;
        let mut worst_fit_waste = 0;

        for (i, bin) in bins.iter().enumerate() {
            let remaining_capacity = bin.capacity.saturating_sub(bin.used_qubits);
            
            if remaining_capacity >= job.requirements.qubits_needed {
                let waste = remaining_capacity - job.requirements.qubits_needed;
                if waste > worst_fit_waste {
                    worst_fit_waste = waste;
                    worst_fit_idx = Some(i);
                }
            }
        }

        Ok(worst_fit_idx)
    }

    /// Create assignment in an existing bin
    fn create_assignment_in_bin(
        &self,
        job: &Job,
        bin: &BinState,
        tile_finder: &TileFinder,
    ) -> Result<Assignment> {
        // For simplicity, create assignment at bin end time
        self.find_assignment_for_job(job, tile_finder, bin.end_time)
    }
}

impl Default for BinPacker {
    fn default() -> Self {
        Self::new()
    }
}

/// State of a bin in bin packing
#[derive(Debug, Clone)]
struct BinState {
    /// Bin identifier
    id: usize,
    /// Total capacity of the bin
    capacity: usize,
    /// Currently used qubits
    used_qubits: usize,
    /// End time of last job in this bin
    end_time: u64,
}

/// Bin packing optimizer for improving packing efficiency
pub struct BinPackOptimizer;

impl BinPackOptimizer {
    /// Optimize a set of assignments to reduce waste
    pub fn optimize_assignments(
        assignments: Vec<Assignment>,
        topology: &Topology,
    ) -> Result<Vec<Assignment>> {
        // Group assignments by time slots
        let mut time_slots: HashMap<u64, Vec<Assignment>> = HashMap::new();
        
        for assignment in assignments {
            time_slots.entry(assignment.start_time)
                .or_insert_with(Vec::new)
                .push(assignment);
        }

        let mut optimized_assignments = Vec::new();

        // Optimize each time slot
        for (start_time, mut slot_assignments) in time_slots {
            // Sort by resource usage (largest first)
            slot_assignments.sort_by(|a, b| {
                b.qubit_mapping.len().cmp(&a.qubit_mapping.len())
            });

            // Try to pack more efficiently
            let optimized_slot = Self::optimize_time_slot(slot_assignments, topology)?;
            optimized_assignments.extend(optimized_slot);
        }

        Ok(optimized_assignments)
    }

    /// Optimize assignments within a single time slot
    fn optimize_time_slot(
        assignments: Vec<Assignment>,
        topology: &Topology,
    ) -> Result<Vec<Assignment>> {
        // For now, return assignments as-is
        // In practice, you'd implement sophisticated packing optimization
        Ok(assignments)
    }

    /// Calculate packing efficiency
    pub fn calculate_efficiency(assignments: &[Assignment], topology: &Topology) -> f64 {
        if assignments.is_empty() {
            return 1.0;
        }

        let total_qubits = topology.qubit_count();
        let mut used_qubits: std::collections::HashSet<usize> = std::collections::HashSet::new();

        for assignment in assignments {
            used_qubits.extend(&assignment.qubit_mapping);
        }

        used_qubits.len() as f64 / total_qubits as f64
    }
}

/// Bin packing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinPackStats {
    /// Number of bins used
    pub bins_used: usize,
    /// Average bin utilization
    pub avg_utilization: f64,
    /// Packing efficiency
    pub efficiency: f64,
    /// Total waste
    pub total_waste: usize,
}

impl BinPackStats {
    /// Calculate statistics from assignments
    pub fn from_assignments(assignments: &[Assignment], topology: &Topology) -> Self {
        let efficiency = BinPackOptimizer::calculate_efficiency(assignments, topology);
        
        // Group by time slots to count bins
        let mut time_slots: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for assignment in assignments {
            time_slots.insert(assignment.start_time);
        }

        let bins_used = time_slots.len();
        let avg_utilization = if bins_used > 0 { efficiency } else { 0.0 };

        let total_assigned_qubits: usize = assignments.iter()
            .map(|a| a.qubit_mapping.len())
            .sum();
        let total_waste = (bins_used * topology.qubit_count()).saturating_sub(total_assigned_qubits);

        Self {
            bins_used,
            avg_utilization,
            efficiency,
            total_waste,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;
    use crate::circuit_ir::CircuitBuilder;
    use crate::scheduler::Job;

    #[test]
    fn test_bin_packer_creation() {
        let packer = BinPacker::new();
        assert_eq!(packer.algorithm, BinPackingAlgorithm::FirstFitDecreasing);
    }

    #[tokio::test]
    async fn test_first_fit_decreasing() {
        let topology = TopologyBuilder::grid(3, 3);
        let packer = BinPacker::new();

        let circuit1 = CircuitBuilder::new("test1", 2, 2).h(0).unwrap().build();
        let circuit2 = CircuitBuilder::new("test2", 3, 3).h(0).unwrap().build();
        
        let jobs = vec![
            Job::new(0, circuit1),
            Job::new(1, circuit2),
        ];

        let batch = Batch {
            id: 0,
            jobs,
            metadata: Default::default(),
        };

        let assignments = packer.pack_batch(batch, &topology).await.unwrap();
        assert_eq!(assignments.len(), 2);
    }

    #[test]
    fn test_bin_pack_stats() {
        let topology = TopologyBuilder::grid(2, 2);
        let assignments = vec![
            Assignment {
                job_id: 0,
                tile_id: 0,
                start_time: 0,
                duration: 1000,
                qubit_mapping: vec![0, 1],
                classical_mapping: vec![0, 1],
                resource_allocation: Default::default(),
            }
        ];

        let stats = BinPackStats::from_assignments(&assignments, &topology);
        assert_eq!(stats.bins_used, 1);
        assert_eq!(stats.efficiency, 0.5); // 2 qubits used out of 4
    }
}