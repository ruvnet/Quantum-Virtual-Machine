//! Assignment data structures and management

use crate::{QvmError, Result, Qubit, ClassicalBit};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Assignment of a job to specific hardware resources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assignment {
    /// Job identifier
    pub job_id: usize,
    /// Assigned tile identifier
    pub tile_id: usize,
    /// Start execution time (microseconds since epoch)
    pub start_time: u64,
    /// Expected execution duration (microseconds)
    pub duration: u64,
    /// Mapping from logical to physical qubits
    pub qubit_mapping: Vec<usize>,
    /// Mapping from logical to physical classical bits
    pub classical_mapping: Vec<usize>,
    /// Additional resource allocations
    pub resource_allocation: ResourceAllocation,
}

/// Additional resource allocation information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Reserved buffer qubits
    pub buffer_qubits: Vec<usize>,
    /// Allocated classical memory
    pub classical_memory: usize,
    /// Special resource allocations
    pub special_resources: HashMap<String, String>,
    /// Quality of service parameters
    pub qos_params: QoSParameters,
}

/// Quality of Service parameters
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QoSParameters {
    /// Expected fidelity
    pub expected_fidelity: f64,
    /// Maximum allowed crosstalk
    pub max_crosstalk: f64,
    /// Coherence time requirements
    pub coherence_requirements: (f64, f64), // (T1, T2)
    /// Error correction overhead
    pub error_correction_overhead: f64,
}

impl Assignment {
    /// Create a new assignment
    pub fn new(
        job_id: usize,
        tile_id: usize,
        start_time: u64,
        duration: u64,
    ) -> Self {
        Self {
            job_id,
            tile_id,
            start_time,
            duration,
            qubit_mapping: Vec::new(),
            classical_mapping: Vec::new(),
            resource_allocation: ResourceAllocation::default(),
        }
    }

    /// Set qubit mapping
    pub fn with_qubit_mapping(mut self, mapping: Vec<usize>) -> Self {
        self.qubit_mapping = mapping;
        self
    }

    /// Set classical mapping
    pub fn with_classical_mapping(mut self, mapping: Vec<usize>) -> Self {
        self.classical_mapping = mapping;
        self
    }

    /// Set resource allocation
    pub fn with_resource_allocation(mut self, allocation: ResourceAllocation) -> Self {
        self.resource_allocation = allocation;
        self
    }

    /// Get end time of the assignment
    pub fn end_time(&self) -> u64 {
        self.start_time + self.duration
    }

    /// Check if this assignment overlaps with another in time
    pub fn overlaps_in_time(&self, other: &Assignment) -> bool {
        !(self.end_time() <= other.start_time || other.end_time() <= self.start_time)
    }

    /// Check if this assignment conflicts with another (time + resources)
    pub fn conflicts_with(&self, other: &Assignment) -> bool {
        if !self.overlaps_in_time(other) {
            return false;
        }

        // Check qubit conflicts
        let self_qubits: std::collections::HashSet<_> = self.qubit_mapping.iter().collect();
        let other_qubits: std::collections::HashSet<_> = other.qubit_mapping.iter().collect();
        
        !self_qubits.is_disjoint(&other_qubits)
    }

    /// Get all physical qubits used (including buffers)
    pub fn all_used_qubits(&self) -> Vec<usize> {
        let mut qubits = self.qubit_mapping.clone();
        qubits.extend(&self.resource_allocation.buffer_qubits);
        qubits.sort();
        qubits.dedup();
        qubits
    }

    /// Validate the assignment
    pub fn validate(&self) -> Result<()> {
        // Check for duplicate qubit mappings
        let mut seen_qubits = std::collections::HashSet::new();
        for &qubit in &self.qubit_mapping {
            if !seen_qubits.insert(qubit) {
                return Err(QvmError::allocation_error(
                    format!("Duplicate qubit {} in assignment", qubit)
                ));
            }
        }

        // Check for duplicate classical mappings
        let mut seen_classical = std::collections::HashSet::new();
        for &classical in &self.classical_mapping {
            if !seen_classical.insert(classical) {
                return Err(QvmError::allocation_error(
                    format!("Duplicate classical bit {} in assignment", classical)
                ));
            }
        }

        // Validate timing
        if self.duration == 0 {
            return Err(QvmError::allocation_error("Assignment duration cannot be zero".to_string()));
        }

        Ok(())
    }

    /// Calculate resource utilization for this assignment
    pub fn resource_utilization(&self, total_qubits: usize) -> AssignmentUtilization {
        let qubit_usage = self.qubit_mapping.len() as f64 / total_qubits as f64;
        let buffer_overhead = self.resource_allocation.buffer_qubits.len() as f64 / total_qubits as f64;
        
        AssignmentUtilization {
            qubit_utilization: qubit_usage,
            buffer_overhead,
            efficiency: qubit_usage / (qubit_usage + buffer_overhead + f64::EPSILON),
            total_resource_usage: qubit_usage + buffer_overhead,
        }
    }
}

/// Assignment utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssignmentUtilization {
    /// Fraction of qubits used for computation
    pub qubit_utilization: f64,
    /// Fraction of qubits used for buffering
    pub buffer_overhead: f64,
    /// Efficiency (computation / total usage)
    pub efficiency: f64,
    /// Total resource usage
    pub total_resource_usage: f64,
}

/// Manager for tracking and optimizing assignments
#[derive(Debug, Clone)]
pub struct AssignmentManager {
    assignments: Vec<Assignment>,
    /// Index by time slot for quick lookups
    time_index: std::collections::BTreeMap<u64, Vec<usize>>,
}

impl AssignmentManager {
    /// Create a new assignment manager
    pub fn new() -> Self {
        Self {
            assignments: Vec::new(),
            time_index: std::collections::BTreeMap::new(),
        }
    }

    /// Add an assignment
    pub fn add_assignment(&mut self, assignment: Assignment) -> Result<()> {
        assignment.validate()?;

        // Check for conflicts with existing assignments
        for existing in &self.assignments {
            if assignment.conflicts_with(existing) {
                return Err(QvmError::allocation_error(
                    format!("Assignment conflicts with existing assignment for job {}", existing.job_id)
                ));
            }
        }

        let assignment_id = self.assignments.len();
        self.assignments.push(assignment.clone());

        // Update time index
        self.time_index
            .entry(assignment.start_time)
            .or_insert_with(Vec::new)
            .push(assignment_id);

        Ok(())
    }

    /// Remove an assignment by job ID
    pub fn remove_assignment(&mut self, job_id: usize) -> Option<Assignment> {
        if let Some(pos) = self.assignments.iter().position(|a| a.job_id == job_id) {
            let removed = self.assignments.remove(pos);
            
            // Update time index
            if let Some(time_assignments) = self.time_index.get_mut(&removed.start_time) {
                time_assignments.retain(|&idx| idx != pos);
                if time_assignments.is_empty() {
                    self.time_index.remove(&removed.start_time);
                }
            }

            // Update indices after removal
            for time_assignments in self.time_index.values_mut() {
                for idx in time_assignments.iter_mut() {
                    if *idx > pos {
                        *idx -= 1;
                    }
                }
            }

            Some(removed)
        } else {
            None
        }
    }

    /// Get assignments active at a specific time
    pub fn assignments_at_time(&self, time: u64) -> Vec<&Assignment> {
        self.assignments
            .iter()
            .filter(|a| a.start_time <= time && time < a.end_time())
            .collect()
    }

    /// Get assignments in a time range
    pub fn assignments_in_range(&self, start: u64, end: u64) -> Vec<&Assignment> {
        self.assignments
            .iter()
            .filter(|a| a.start_time < end && a.end_time() > start)
            .collect()
    }

    /// Find next available time slot for a job
    pub fn find_next_available_time(&self, duration: u64, required_qubits: &[usize]) -> u64 {
        let mut time = 0;
        let time_step = 1000; // 1ms granularity

        loop {
            let active_assignments = self.assignments_at_time(time);
            let used_qubits: std::collections::HashSet<usize> = active_assignments
                .iter()
                .flat_map(|a| a.all_used_qubits())
                .collect();

            // Check if all required qubits are available
            if required_qubits.iter().all(|&qubit| !used_qubits.contains(&qubit)) {
                return time;
            }

            time += time_step;
            
            // Safety check to avoid infinite loops
            if time > 1_000_000_000 { // 1000 seconds
                break;
            }
        }

        time
    }

    /// Optimize assignments to reduce conflicts and improve utilization
    pub fn optimize_assignments(&mut self) -> Result<()> {
        // Sort assignments by start time
        self.assignments.sort_by_key(|a| a.start_time);
        
        // Rebuild time index
        self.time_index.clear();
        for (i, assignment) in self.assignments.iter().enumerate() {
            self.time_index
                .entry(assignment.start_time)
                .or_insert_with(Vec::new)
                .push(i);
        }

        // Try to compress timeline by moving assignments earlier when possible
        for i in 0..self.assignments.len() {
            let current_assignment = &self.assignments[i];
            let required_qubits = current_assignment.all_used_qubits();
            
            let earliest_start = self.find_earliest_start_time(
                current_assignment.duration,
                &required_qubits,
                i
            );

            if earliest_start < current_assignment.start_time {
                // Move assignment to earlier time
                let mut updated_assignment = current_assignment.clone();
                updated_assignment.start_time = earliest_start;
                self.assignments[i] = updated_assignment;
            }
        }

        // Rebuild time index after optimization
        self.time_index.clear();
        for (i, assignment) in self.assignments.iter().enumerate() {
            self.time_index
                .entry(assignment.start_time)
                .or_insert_with(Vec::new)
                .push(i);
        }

        Ok(())
    }

    /// Find earliest start time for an assignment, excluding a specific index
    fn find_earliest_start_time(
        &self,
        duration: u64,
        required_qubits: &[usize],
        exclude_idx: usize,
    ) -> u64 {
        let mut time = 0;
        let time_step = 1000;

        loop {
            let active_assignments: Vec<&Assignment> = self.assignments
                .iter()
                .enumerate()
                .filter(|(i, a)| *i != exclude_idx && a.start_time <= time && time < a.end_time())
                .map(|(_, a)| a)
                .collect();

            let used_qubits: std::collections::HashSet<usize> = active_assignments
                .iter()
                .flat_map(|a| a.all_used_qubits())
                .collect();

            if required_qubits.iter().all(|&qubit| !used_qubits.contains(&qubit)) {
                return time;
            }

            time += time_step;

            if time > 1_000_000_000 {
                break;
            }
        }

        time
    }

    /// Get all assignments
    pub fn assignments(&self) -> &[Assignment] {
        &self.assignments
    }

    /// Get assignment count
    pub fn count(&self) -> usize {
        self.assignments.len()
    }

    /// Clear all assignments
    pub fn clear(&mut self) {
        self.assignments.clear();
        self.time_index.clear();
    }

    /// Calculate overall utilization statistics
    pub fn utilization_stats(&self, total_qubits: usize) -> OverallUtilization {
        if self.assignments.is_empty() {
            return OverallUtilization::default();
        }

        let max_time = self.assignments
            .iter()
            .map(|a| a.end_time())
            .max()
            .unwrap_or(0);

        let mut total_qubit_time = 0u64;
        let mut total_buffer_time = 0u64;

        for assignment in &self.assignments {
            total_qubit_time += assignment.qubit_mapping.len() as u64 * assignment.duration;
            total_buffer_time += assignment.resource_allocation.buffer_qubits.len() as u64 * assignment.duration;
        }

        let total_possible_time = total_qubits as u64 * max_time;
        let utilization = if total_possible_time > 0 {
            total_qubit_time as f64 / total_possible_time as f64
        } else {
            0.0
        };

        let buffer_overhead = if total_possible_time > 0 {
            total_buffer_time as f64 / total_possible_time as f64
        } else {
            0.0
        };

        OverallUtilization {
            average_utilization: utilization,
            peak_utilization: 0.0, // Would need time-series analysis
            buffer_overhead,
            efficiency: utilization / (utilization + buffer_overhead + f64::EPSILON),
            total_assignments: self.assignments.len(),
        }
    }
}

impl Default for AssignmentManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Overall utilization statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OverallUtilization {
    /// Average resource utilization
    pub average_utilization: f64,
    /// Peak resource utilization
    pub peak_utilization: f64,
    /// Buffer overhead
    pub buffer_overhead: f64,
    /// Overall efficiency
    pub efficiency: f64,
    /// Total number of assignments
    pub total_assignments: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assignment_creation() {
        let assignment = Assignment::new(0, 1, 1000, 5000)
            .with_qubit_mapping(vec![0, 1, 2])
            .with_classical_mapping(vec![0, 1]);

        assert_eq!(assignment.job_id, 0);
        assert_eq!(assignment.tile_id, 1);
        assert_eq!(assignment.end_time(), 6000);
        assert_eq!(assignment.qubit_mapping.len(), 3);
    }

    #[test]
    fn test_assignment_conflicts() {
        let assignment1 = Assignment::new(0, 1, 1000, 2000)
            .with_qubit_mapping(vec![0, 1]);
        let assignment2 = Assignment::new(1, 2, 1500, 2000)
            .with_qubit_mapping(vec![1, 2]);

        assert!(assignment1.conflicts_with(&assignment2));
    }

    #[test]
    fn test_assignment_manager() {
        let mut manager = AssignmentManager::new();
        
        let assignment = Assignment::new(0, 1, 1000, 2000)
            .with_qubit_mapping(vec![0, 1]);

        manager.add_assignment(assignment).unwrap();
        assert_eq!(manager.count(), 1);

        let active = manager.assignments_at_time(1500);
        assert_eq!(active.len(), 1);

        let removed = manager.remove_assignment(0);
        assert!(removed.is_some());
        assert_eq!(manager.count(), 0);
    }

    #[test]
    fn test_next_available_time() {
        let mut manager = AssignmentManager::new();
        
        let assignment = Assignment::new(0, 1, 1000, 2000)
            .with_qubit_mapping(vec![0, 1]);
        manager.add_assignment(assignment).unwrap();

        // Should find time after existing assignment
        let next_time = manager.find_next_available_time(1000, &[0]);
        assert!(next_time >= 3000);

        // Different qubits should be available immediately
        let next_time = manager.find_next_available_time(1000, &[2, 3]);
        assert_eq!(next_time, 0);
    }
}