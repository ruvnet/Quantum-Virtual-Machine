//! Qubit and classical bit mapping management

use crate::{QvmError, Result, Qubit, ClassicalBit, Topology};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Resource mapping for a circuit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMapping {
    /// Logical to physical qubit mapping
    pub qubit_mapping: Vec<usize>,
    /// Logical to physical classical bit mapping
    pub classical_mapping: Vec<usize>,
    /// Mapping metadata
    pub metadata: MappingMetadata,
}

/// Mapping metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MappingMetadata {
    /// Mapping quality score (0.0 to 1.0)
    pub quality_score: f64,
    /// Number of SWAP operations needed
    pub swap_count: usize,
    /// Routing overhead
    pub routing_overhead: f64,
    /// Custom mapping properties
    pub properties: HashMap<String, String>,
}

/// Qubit mapper for creating optimal qubit assignments
#[derive(Debug, Clone)]
pub struct QubitMapper {
    topology: Topology,
    config: MapperConfig,
}

/// Mapper configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperConfig {
    /// Mapping strategy
    pub strategy: MappingStrategy,
    /// Enable SWAP insertion
    pub enable_swaps: bool,
    /// Maximum SWAP overhead allowed
    pub max_swap_overhead: f64,
    /// Prioritize connectivity
    pub prioritize_connectivity: bool,
}

impl Default for MapperConfig {
    fn default() -> Self {
        Self {
            strategy: MappingStrategy::ConnectivityAware,
            enable_swaps: true,
            max_swap_overhead: 0.5,
            prioritize_connectivity: true,
        }
    }
}

/// Mapping strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MappingStrategy {
    /// Simple linear mapping
    Linear,
    /// Random mapping
    Random,
    /// Connectivity-aware mapping
    ConnectivityAware,
    /// Distance-minimizing mapping
    DistanceMinimizing,
    /// Adaptive mapping
    Adaptive,
}

impl QubitMapper {
    /// Create a new qubit mapper
    pub fn new(topology: &Topology) -> Self {
        Self {
            topology: topology.clone(),
            config: MapperConfig::default(),
        }
    }

    /// Create a classical bit allocator
    pub fn create_classical_allocator(&self) -> ClassicalAllocator {
        ClassicalAllocator::new(self.topology.qubit_count()) // Assume same number of classical bits as qubits
    }

    /// Create mapper with custom configuration
    pub fn with_config(topology: &Topology, config: MapperConfig) -> Self {
        Self {
            topology: topology.clone(),
            config,
        }
    }

    /// Create a resource mapping
    pub fn create_mapping(
        &self,
        qubit_assignments: &[usize],
        classical_assignments: &[usize],
    ) -> Result<ResourceMapping> {
        let quality_score = self.calculate_mapping_quality(qubit_assignments);
        
        let metadata = MappingMetadata {
            quality_score,
            swap_count: 0, // Would be calculated during routing
            routing_overhead: 0.0,
            properties: HashMap::new(),
        };

        Ok(ResourceMapping {
            qubit_mapping: qubit_assignments.to_vec(),
            classical_mapping: classical_assignments.to_vec(),
            metadata,
        })
    }

    /// Find optimal qubit mapping for a circuit
    pub fn find_optimal_mapping(
        &self,
        logical_qubits: usize,
        connectivity_requirements: &[(usize, usize)],
    ) -> Result<Vec<usize>> {
        match self.config.strategy {
            MappingStrategy::Linear => self.linear_mapping(logical_qubits),
            MappingStrategy::Random => self.random_mapping(logical_qubits),
            MappingStrategy::ConnectivityAware => {
                self.connectivity_aware_mapping(logical_qubits, connectivity_requirements)
            }
            MappingStrategy::DistanceMinimizing => {
                self.distance_minimizing_mapping(logical_qubits, connectivity_requirements)
            }
            MappingStrategy::Adaptive => {
                self.adaptive_mapping(logical_qubits, connectivity_requirements)
            }
        }
    }

    /// Linear mapping strategy
    fn linear_mapping(&self, logical_qubits: usize) -> Result<Vec<usize>> {
        if logical_qubits > self.topology.qubit_count() {
            return Err(QvmError::allocation_error(
                "Not enough physical qubits available".to_string()
            ));
        }

        Ok((0..logical_qubits).collect())
    }

    /// Random mapping strategy
    fn random_mapping(&self, logical_qubits: usize) -> Result<Vec<usize>> {
        use std::collections::HashSet;

        if logical_qubits > self.topology.qubit_count() {
            return Err(QvmError::allocation_error(
                "Not enough physical qubits available".to_string()
            ));
        }

        let mut mapping = Vec::new();
        let mut used_qubits = HashSet::new();
        
        // Simple pseudo-random assignment (not cryptographically secure)
        let mut seed = 12345u64;
        for _ in 0..logical_qubits {
            loop {
                seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
                let physical_qubit = (seed as usize) % self.topology.qubit_count();
                
                if !used_qubits.contains(&physical_qubit) {
                    mapping.push(physical_qubit);
                    used_qubits.insert(physical_qubit);
                    break;
                }
            }
        }

        Ok(mapping)
    }

    /// Connectivity-aware mapping strategy
    fn connectivity_aware_mapping(
        &self,
        logical_qubits: usize,
        connectivity_requirements: &[(usize, usize)],
    ) -> Result<Vec<usize>> {
        if logical_qubits > self.topology.qubit_count() {
            return Err(QvmError::allocation_error(
                "Not enough physical qubits available".to_string()
            ));
        }

        // Start with linear mapping
        let mut mapping = (0..logical_qubits).collect::<Vec<_>>();

        // Try to improve mapping based on connectivity requirements
        for &(logical1, logical2) in connectivity_requirements {
            if logical1 >= logical_qubits || logical2 >= logical_qubits {
                continue;
            }

            let physical1 = mapping[logical1];
            let physical2 = mapping[logical2];

            // Check if these physical qubits are connected
            if !self.topology.are_connected(Qubit(physical1), Qubit(physical2)) {
                // Try to find better mapping
                if let Some(better_mapping) = self.find_connected_pair(logical1, logical2, &mapping) {
                    mapping = better_mapping;
                }
            }
        }

        Ok(mapping)
    }

    /// Distance-minimizing mapping strategy
    fn distance_minimizing_mapping(
        &self,
        logical_qubits: usize,
        connectivity_requirements: &[(usize, usize)],
    ) -> Result<Vec<usize>> {
        if logical_qubits > self.topology.qubit_count() {
            return Err(QvmError::allocation_error(
                "Not enough physical qubits available".to_string()
            ));
        }

        // Simple greedy approach: place the most connected qubits on central qubits
        let mut mapping = vec![0; logical_qubits];
        let mut used_physical = std::collections::HashSet::new();

        // Count logical qubit connectivity
        let mut connectivity_count = vec![0; logical_qubits];
        for &(q1, q2) in connectivity_requirements {
            if q1 < logical_qubits { connectivity_count[q1] += 1; }
            if q2 < logical_qubits { connectivity_count[q2] += 1; }
        }

        // Sort by connectivity (most connected first)
        let mut qubit_order: Vec<_> = (0..logical_qubits).collect();
        qubit_order.sort_by_key(|&q| std::cmp::Reverse(connectivity_count[q]));

        // Assign most connected qubits to most central physical qubits
        let physical_centrality = self.calculate_centrality();
        let mut physical_order: Vec<_> = (0..self.topology.qubit_count()).collect();
        physical_order.sort_by(|&a, &b| {
            physical_centrality[b].partial_cmp(&physical_centrality[a]).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (i, &logical_qubit) in qubit_order.iter().enumerate() {
            if i < physical_order.len() {
                mapping[logical_qubit] = physical_order[i];
                used_physical.insert(physical_order[i]);
            }
        }

        Ok(mapping)
    }

    /// Adaptive mapping strategy
    fn adaptive_mapping(
        &self,
        logical_qubits: usize,
        connectivity_requirements: &[(usize, usize)],
    ) -> Result<Vec<usize>> {
        // Choose strategy based on circuit characteristics
        let connectivity_density = connectivity_requirements.len() as f64 / (logical_qubits * logical_qubits) as f64;
        
        if connectivity_density > 0.5 {
            self.distance_minimizing_mapping(logical_qubits, connectivity_requirements)
        } else if connectivity_density > 0.1 {
            self.connectivity_aware_mapping(logical_qubits, connectivity_requirements)
        } else {
            self.linear_mapping(logical_qubits)
        }
    }

    /// Calculate centrality of each physical qubit
    fn calculate_centrality(&self) -> Vec<f64> {
        let qubit_count = self.topology.qubit_count();
        let mut centrality = vec![0.0; qubit_count];

        for i in 0..qubit_count {
            // Simple centrality: number of neighbors
            centrality[i] = self.topology.neighbors(Qubit(i)).len() as f64;
        }

        centrality
    }

    /// Find a connected pair for two logical qubits
    fn find_connected_pair(
        &self,
        logical1: usize,
        logical2: usize,
        current_mapping: &[usize],
    ) -> Option<Vec<usize>> {
        let mut used_physical: std::collections::HashSet<_> = current_mapping.iter().collect();
        
        // Try different physical qubit pairs
        for physical1 in 0..self.topology.qubit_count() {
            if used_physical.contains(&physical1) && current_mapping[logical1] != physical1 {
                continue;
            }

            for neighbor in self.topology.neighbors(Qubit(physical1)) {
                let physical2 = neighbor.index();
                
                if used_physical.contains(&physical2) && current_mapping[logical2] != physical2 {
                    continue;
                }

                // Create new mapping
                let mut new_mapping = current_mapping.to_vec();
                new_mapping[logical1] = physical1;
                new_mapping[logical2] = physical2;
                
                return Some(new_mapping);
            }
        }

        None
    }

    /// Calculate mapping quality score
    fn calculate_mapping_quality(&self, mapping: &[usize]) -> f64 {
        if mapping.is_empty() {
            return 1.0;
        }

        let mut total_distance = 0.0;
        let mut pair_count = 0;

        // Calculate average distance between consecutive qubits
        for i in 0..(mapping.len() - 1) {
            if let Some(path) = self.topology.shortest_path(Qubit(mapping[i]), Qubit(mapping[i + 1])) {
                total_distance += (path.len() - 1) as f64;
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            return 1.0;
        }

        let avg_distance = total_distance / pair_count as f64;
        
        // Quality decreases with distance (1.0 for distance 1, 0.5 for distance 2, etc.)
        1.0 / (1.0 + avg_distance - 1.0)
    }

    /// Route a two-qubit gate with SWAP insertion if needed
    pub fn route_two_qubit_gate(
        &self,
        control: usize,
        target: usize,
        mapping: &mut [usize],
    ) -> Result<Vec<SwapOperation>> {
        let physical_control = mapping[control];
        let physical_target = mapping[target];

        if self.topology.are_connected(Qubit(physical_control), Qubit(physical_target)) {
            // Already connected, no SWAPs needed
            return Ok(vec![]);
        }

        if !self.config.enable_swaps {
            return Err(QvmError::allocation_error(
                "Qubits not connected and SWAP insertion disabled".to_string()
            ));
        }

        // Find shortest path and insert SWAPs
        if let Some(path) = self.topology.shortest_path(Qubit(physical_control), Qubit(physical_target)) {
            let mut swaps = Vec::new();
            
            // Insert SWAPs to bring qubits together
            for i in 0..(path.len() - 2) {
                let swap_op = SwapOperation {
                    qubit1: path[i].index(),
                    qubit2: path[i + 1].index(),
                };
                swaps.push(swap_op);
                
                // Update mapping
                // This is simplified - in practice you'd need more sophisticated tracking
            }

            // Check swap overhead
            let overhead = swaps.len() as f64 / mapping.len() as f64;
            if overhead > self.config.max_swap_overhead {
                return Err(QvmError::allocation_error(
                    "SWAP overhead too high".to_string()
                ));
            }

            Ok(swaps)
        } else {
            Err(QvmError::allocation_error(
                "No path found between qubits".to_string()
            ))
        }
    }
}

/// SWAP operation for routing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapOperation {
    pub qubit1: usize,
    pub qubit2: usize,
}

/// Classical bit allocator for managing classical bit assignments
#[derive(Debug, Clone)]
pub struct ClassicalAllocator {
    /// Total classical bits available
    total_bits: usize,
    /// Currently allocated bits
    allocated_bits: std::collections::HashSet<usize>,
    /// Allocation strategy
    strategy: ClassicalAllocationStrategy,
}

/// Classical bit allocation strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassicalAllocationStrategy {
    /// Sequential allocation
    Sequential,
    /// Random allocation
    Random,
    /// Minimize conflicts with measurement patterns
    ConflictMinimizing,
}

impl ClassicalAllocator {
    /// Create a new classical allocator
    pub fn new(total_bits: usize) -> Self {
        Self {
            total_bits,
            allocated_bits: std::collections::HashSet::new(),
            strategy: ClassicalAllocationStrategy::Sequential,
        }
    }

    /// Allocate classical bits for a circuit
    pub fn allocate_bits(&mut self, required_bits: usize) -> Result<Vec<usize>> {
        if required_bits > self.available_bits() {
            return Err(QvmError::allocation_error(
                format!("Not enough classical bits: need {}, have {}", 
                       required_bits, self.available_bits())
            ));
        }

        match self.strategy {
            ClassicalAllocationStrategy::Sequential => self.allocate_sequential(required_bits),
            ClassicalAllocationStrategy::Random => self.allocate_random(required_bits),
            ClassicalAllocationStrategy::ConflictMinimizing => self.allocate_conflict_minimizing(required_bits),
        }
    }

    /// Sequential allocation
    fn allocate_sequential(&mut self, required_bits: usize) -> Result<Vec<usize>> {
        let mut allocation = Vec::new();
        
        for bit in 0..self.total_bits {
            if !self.allocated_bits.contains(&bit) {
                allocation.push(bit);
                self.allocated_bits.insert(bit);
                
                if allocation.len() >= required_bits {
                    break;
                }
            }
        }
        
        Ok(allocation)
    }

    /// Random allocation
    fn allocate_random(&mut self, required_bits: usize) -> Result<Vec<usize>> {
        let mut allocation = Vec::new();
        let available: Vec<_> = (0..self.total_bits)
            .filter(|bit| !self.allocated_bits.contains(bit))
            .collect();
        
        // Simple pseudo-random selection
        let mut seed = 54321u64;
        for _ in 0..required_bits {
            if available.is_empty() {
                break;
            }
            
            seed = (seed.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
            let idx = (seed as usize) % available.len();
            let bit = available[idx];
            
            allocation.push(bit);
            self.allocated_bits.insert(bit);
        }
        
        Ok(allocation)
    }

    /// Conflict-minimizing allocation
    fn allocate_conflict_minimizing(&mut self, required_bits: usize) -> Result<Vec<usize>> {
        // For now, use sequential allocation
        // In a real implementation, this would analyze measurement patterns
        self.allocate_sequential(required_bits)
    }

    /// Release allocated bits
    pub fn release_bits(&mut self, bits: &[usize]) {
        for &bit in bits {
            self.allocated_bits.remove(&bit);
        }
    }

    /// Get number of available bits
    pub fn available_bits(&self) -> usize {
        self.total_bits - self.allocated_bits.len()
    }

    /// Reset all allocations
    pub fn reset(&mut self) {
        self.allocated_bits.clear();
    }

    /// Check if a bit is allocated
    pub fn is_allocated(&self, bit: usize) -> bool {
        self.allocated_bits.contains(&bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;

    #[test]
    fn test_qubit_mapper_creation() {
        let topology = TopologyBuilder::grid(3, 3);
        let mapper = QubitMapper::new(&topology);
        assert_eq!(mapper.config.strategy, MappingStrategy::ConnectivityAware);
    }

    #[test]
    fn test_linear_mapping() {
        let topology = TopologyBuilder::grid(3, 3);
        let mapper = QubitMapper::new(&topology);
        
        let mapping = mapper.linear_mapping(4).unwrap();
        assert_eq!(mapping, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_mapping_quality() {
        let topology = TopologyBuilder::linear(5);
        let mapper = QubitMapper::new(&topology);
        
        let good_mapping = vec![0, 1, 2]; // Sequential
        let bad_mapping = vec![0, 2, 4];  // Gaps
        
        let quality_good = mapper.calculate_mapping_quality(&good_mapping);
        let quality_bad = mapper.calculate_mapping_quality(&bad_mapping);
        
        assert!(quality_good > quality_bad);
    }

    #[test]
    fn test_resource_mapping_creation() {
        let topology = TopologyBuilder::grid(2, 2);
        let mapper = QubitMapper::new(&topology);
        
        let qubit_assignments = vec![0, 1];
        let classical_assignments = vec![0, 1];
        
        let mapping = mapper.create_mapping(&qubit_assignments, &classical_assignments).unwrap();
        assert_eq!(mapping.qubit_mapping, vec![0, 1]);
        assert_eq!(mapping.classical_mapping, vec![0, 1]);
        assert!(mapping.metadata.quality_score > 0.0);
    }

    #[test]
    fn test_connectivity_aware_mapping() {
        let topology = TopologyBuilder::grid(3, 3);
        let mapper = QubitMapper::new(&topology);
        
        let connectivity_requirements = vec![(0, 1), (1, 2)];
        let mapping = mapper.connectivity_aware_mapping(3, &connectivity_requirements).unwrap();
        
        assert_eq!(mapping.len(), 3);
        // Should try to keep connected qubits close
    }
}