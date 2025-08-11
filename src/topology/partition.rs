//! Topology partitioning algorithms

use crate::{QvmError, Result, Qubit};
use crate::topology::{Topology, Tile, TileBounds, TileFinder};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Partitioning strategy for topology division
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Grid-based rectangular partitioning
    Rectangular,
    /// Graph-based partitioning (minimizing cuts)
    GraphBased,
    /// Uniform size partitioning
    Uniform,
    /// Custom partitioning based on user-defined regions
    Custom,
}

/// Partition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionConfig {
    /// Partitioning strategy
    pub strategy: PartitionStrategy,
    /// Target number of partitions
    pub target_partitions: usize,
    /// Minimum qubits per partition
    pub min_qubits_per_partition: usize,
    /// Maximum qubits per partition
    pub max_qubits_per_partition: usize,
    /// Buffer size between partitions
    pub buffer_size: usize,
    /// Balance factor (0.0 = allow imbalanced, 1.0 = perfect balance)
    pub balance_factor: f64,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            strategy: PartitionStrategy::Rectangular,
            target_partitions: 4,
            min_qubits_per_partition: 2,
            max_qubits_per_partition: 20,
            buffer_size: 1,
            balance_factor: 0.8,
        }
    }
}

/// Result of topology partitioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionResult {
    /// Generated tiles/partitions
    pub tiles: Vec<Tile>,
    /// Partitioning statistics
    pub statistics: PartitionStatistics,
}

/// Statistics about the partitioning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionStatistics {
    /// Total number of partitions
    pub partition_count: usize,
    /// Average qubits per partition
    pub avg_qubits_per_partition: f64,
    /// Standard deviation of partition sizes
    pub size_std_deviation: f64,
    /// Total edge cuts between partitions
    pub total_edge_cuts: usize,
    /// Partitioning efficiency (0.0 to 1.0)
    pub efficiency: f64,
}

/// Topology partitioner
pub struct TopologyPartitioner<'a> {
    topology: &'a Topology,
}

impl<'a> TopologyPartitioner<'a> {
    /// Create a new partitioner
    pub fn new(topology: &'a Topology) -> Self {
        Self { topology }
    }

    /// Partition the topology according to the configuration
    pub fn partition(&self, config: &PartitionConfig) -> Result<PartitionResult> {
        match config.strategy {
            PartitionStrategy::Rectangular => self.rectangular_partition(config),
            PartitionStrategy::GraphBased => self.graph_based_partition(config),
            PartitionStrategy::Uniform => self.uniform_partition(config),
            PartitionStrategy::Custom => self.custom_partition(config),
        }
    }

    /// Rectangular grid-based partitioning
    fn rectangular_partition(&self, config: &PartitionConfig) -> Result<PartitionResult> {
        let (width, height) = self.topology.metadata().dimensions
            .ok_or_else(|| QvmError::topology_error("Topology dimensions not available for rectangular partitioning".to_string()))?;

        // Calculate partition dimensions
        let partitions_per_row = (config.target_partitions as f64).sqrt().ceil() as usize;
        let partitions_per_col = (config.target_partitions + partitions_per_row - 1) / partitions_per_row;

        let partition_width = width / partitions_per_row;
        let partition_height = height / partitions_per_col;

        let mut tiles = Vec::new();
        let tile_finder = TileFinder::new(self.topology);
        let mut tile_id = 0;

        for row in 0..partitions_per_col {
            for col in 0..partitions_per_row {
                let min_x = (col * partition_width) as i32;
                let max_x = ((col + 1) * partition_width - 1).min(width - 1) as i32;
                let min_y = (row * partition_height) as i32;
                let max_y = ((row + 1) * partition_height - 1).min(height - 1) as i32;

                let bounds = TileBounds::new(min_x, max_x, min_y, max_y);
                
                if let Ok(tile) = tile_finder.create_tile_from_bounds(tile_id, bounds, config.buffer_size) {
                    if tile.qubit_count() >= config.min_qubits_per_partition {
                        tiles.push(tile);
                        tile_id += 1;
                    }
                }
            }
        }

        let statistics = self.calculate_statistics(&tiles, config);
        Ok(PartitionResult { tiles, statistics })
    }

    /// Graph-based partitioning using spectral methods (simplified)
    fn graph_based_partition(&self, config: &PartitionConfig) -> Result<PartitionResult> {
        // This is a simplified implementation
        // In practice, you'd use sophisticated graph partitioning algorithms like METIS
        
        let qubits = self.topology.qubits();
        let mut partitions = Vec::new();
        let qubits_per_partition = (qubits.len() + config.target_partitions - 1) / config.target_partitions;

        // Simple round-robin assignment (placeholder for real algorithm)
        for chunk in qubits.chunks(qubits_per_partition) {
            if chunk.len() >= config.min_qubits_per_partition {
                // Find bounding box for these qubits
                let bounds = self.calculate_bounding_box(chunk)?;
                let mut tile = Tile::new(partitions.len(), bounds);
                
                for &qubit in chunk {
                    tile.add_qubit(qubit);
                }
                
                partitions.push(tile);
            }
        }

        let statistics = self.calculate_statistics(&partitions, config);
        Ok(PartitionResult { tiles: partitions, statistics })
    }

    /// Uniform partitioning with balanced sizes
    fn uniform_partition(&self, config: &PartitionConfig) -> Result<PartitionResult> {
        let qubits = self.topology.qubits();
        let target_size = qubits.len() / config.target_partitions;
        
        let mut tiles = Vec::new();
        let mut current_partition = Vec::new();
        let mut tile_id = 0;

        for qubit in qubits {
            current_partition.push(qubit);
            
            if current_partition.len() >= target_size || current_partition.len() >= config.max_qubits_per_partition {
                if current_partition.len() >= config.min_qubits_per_partition {
                    let bounds = self.calculate_bounding_box(&current_partition)?;
                    let mut tile = Tile::new(tile_id, bounds);
                    
                    for &qubit in &current_partition {
                        tile.add_qubit(qubit);
                    }
                    
                    tiles.push(tile);
                    tile_id += 1;
                }
                current_partition.clear();
            }
        }

        // Handle remaining qubits
        if !current_partition.is_empty() && current_partition.len() >= config.min_qubits_per_partition {
            let bounds = self.calculate_bounding_box(&current_partition)?;
            let mut tile = Tile::new(tile_id, bounds);
            
            for &qubit in &current_partition {
                tile.add_qubit(qubit);
            }
            
            tiles.push(tile);
        }

        let statistics = self.calculate_statistics(&tiles, config);
        Ok(PartitionResult { tiles, statistics })
    }

    /// Custom partitioning (placeholder for user-defined strategies)
    fn custom_partition(&self, config: &PartitionConfig) -> Result<PartitionResult> {
        // For now, delegate to rectangular partitioning
        // In practice, this would allow user-defined partitioning functions
        self.rectangular_partition(config)
    }

    /// Calculate bounding box for a set of qubits
    fn calculate_bounding_box(&self, qubits: &[Qubit]) -> Result<TileBounds> {
        let mut min_x = i32::MAX;
        let mut max_x = i32::MIN;
        let mut min_y = i32::MAX;
        let mut max_y = i32::MIN;
        let mut found_position = false;

        for &qubit in qubits {
            if let Some(position) = self.topology.position(qubit) {
                min_x = min_x.min(position.x);
                max_x = max_x.max(position.x);
                min_y = min_y.min(position.y);
                max_y = max_y.max(position.y);
                found_position = true;
            }
        }

        if !found_position {
            return Err(QvmError::topology_error("No position information available for qubits".to_string()));
        }

        Ok(TileBounds::new(min_x, max_x, min_y, max_y))
    }

    /// Calculate partitioning statistics
    fn calculate_statistics(&self, tiles: &[Tile], _config: &PartitionConfig) -> PartitionStatistics {
        let partition_count = tiles.len();
        
        if partition_count == 0 {
            return PartitionStatistics {
                partition_count: 0,
                avg_qubits_per_partition: 0.0,
                size_std_deviation: 0.0,
                total_edge_cuts: 0,
                efficiency: 0.0,
            };
        }

        // Calculate size statistics
        let sizes: Vec<usize> = tiles.iter().map(|t| t.qubit_count()).collect();
        let avg_qubits_per_partition = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;
        
        let variance = sizes.iter()
            .map(|&size| (size as f64 - avg_qubits_per_partition).powi(2))
            .sum::<f64>() / sizes.len() as f64;
        let size_std_deviation = variance.sqrt();

        // Calculate edge cuts (simplified)
        let total_edge_cuts = self.calculate_edge_cuts(tiles);

        // Calculate efficiency (balance vs cuts trade-off)
        let balance_score = 1.0 - (size_std_deviation / avg_qubits_per_partition.max(1.0));
        let cut_penalty = total_edge_cuts as f64 / self.topology.connection_count() as f64;
        let efficiency = (balance_score - cut_penalty * 0.5).max(0.0).min(1.0);

        PartitionStatistics {
            partition_count,
            avg_qubits_per_partition,
            size_std_deviation,
            total_edge_cuts,
            efficiency,
        }
    }

    /// Calculate the number of edges cut by the partitioning
    fn calculate_edge_cuts(&self, tiles: &[Tile]) -> usize {
        let mut qubit_to_partition: HashMap<Qubit, usize> = HashMap::new();
        
        for (i, tile) in tiles.iter().enumerate() {
            for &qubit in &tile.qubits {
                qubit_to_partition.insert(qubit, i);
            }
        }

        let mut cuts = 0;
        for qubit in self.topology.qubits() {
            if let Some(&partition) = qubit_to_partition.get(&qubit) {
                for neighbor in self.topology.neighbors(qubit) {
                    if let Some(&neighbor_partition) = qubit_to_partition.get(&neighbor) {
                        if partition != neighbor_partition {
                            cuts += 1;
                        }
                    }
                }
            }
        }

        // Each cut is counted twice (once from each side)
        cuts / 2
    }
}

/// Partition optimizer for improving existing partitions
pub struct PartitionOptimizer;

impl PartitionOptimizer {
    /// Optimize a partition by minimizing cuts and balancing sizes
    pub fn optimize_partition(
        topology: &Topology,
        mut tiles: Vec<Tile>,
        max_iterations: usize,
    ) -> Result<Vec<Tile>> {
        for _iteration in 0..max_iterations {
            let mut improved = false;
            
            // Try moving qubits between adjacent partitions
            for i in 0..tiles.len() {
                for j in (i + 1)..tiles.len() {
                    let (left, right) = tiles.split_at_mut(j);
                    if Self::try_improve_partition_pair(topology, &mut left[i], &mut right[0]) {
                        improved = true;
                    }
                }
            }
            
            if !improved {
                break;
            }
        }

        Ok(tiles)
    }

    /// Try to improve a pair of partitions by swapping boundary qubits
    fn try_improve_partition_pair(
        topology: &Topology,
        tile1: &mut Tile,
        tile2: &mut Tile,
    ) -> bool {
        // Find boundary qubits (qubits with neighbors in the other partition)
        let tile1_qubits: HashSet<_> = tile1.qubits.iter().copied().collect();
        let tile2_qubits: HashSet<_> = tile2.qubits.iter().copied().collect();

        let mut boundary1 = Vec::new();
        let mut boundary2 = Vec::new();

        for &qubit in &tile1_qubits {
            for neighbor in topology.neighbors(qubit) {
                if tile2_qubits.contains(&neighbor) {
                    boundary1.push(qubit);
                    break;
                }
            }
        }

        for &qubit in &tile2_qubits {
            for neighbor in topology.neighbors(qubit) {
                if tile1_qubits.contains(&neighbor) {
                    boundary2.push(qubit);
                    break;
                }
            }
        }

        // Try swapping boundary qubits to reduce cuts
        for &qubit1 in &boundary1 {
            for &qubit2 in &boundary2 {
                if Self::would_reduce_cuts(topology, qubit1, qubit2, tile1, tile2) {
                    // Perform the swap
                    tile1.remove_qubit(qubit1);
                    tile1.add_qubit(qubit2);
                    tile2.remove_qubit(qubit2);
                    tile2.add_qubit(qubit1);
                    return true;
                }
            }
        }

        false
    }

    /// Check if swapping two qubits would reduce the number of cuts
    fn would_reduce_cuts(
        topology: &Topology,
        qubit1: Qubit,
        qubit2: Qubit,
        tile1: &Tile,
        tile2: &Tile,
    ) -> bool {
        let tile1_qubits: HashSet<_> = tile1.qubits.iter().copied().collect();
        let tile2_qubits: HashSet<_> = tile2.qubits.iter().copied().collect();

        // Calculate current cuts for these qubits
        let mut current_cuts = 0;
        for neighbor in topology.neighbors(qubit1) {
            if tile2_qubits.contains(&neighbor) {
                current_cuts += 1;
            }
        }
        for neighbor in topology.neighbors(qubit2) {
            if tile1_qubits.contains(&neighbor) {
                current_cuts += 1;
            }
        }

        // Calculate cuts after swap
        let mut new_cuts = 0;
        for neighbor in topology.neighbors(qubit1) {
            if tile1_qubits.contains(&neighbor) && neighbor != qubit2 {
                new_cuts += 1;
            }
        }
        for neighbor in topology.neighbors(qubit2) {
            if tile2_qubits.contains(&neighbor) && neighbor != qubit1 {
                new_cuts += 1;
            }
        }

        new_cuts < current_cuts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;

    #[test]
    fn test_rectangular_partition() {
        let topology = TopologyBuilder::grid(4, 4);
        let partitioner = TopologyPartitioner::new(&topology);
        
        let config = PartitionConfig {
            target_partitions: 4,
            min_qubits_per_partition: 2,
            ..Default::default()
        };

        let result = partitioner.partition(&config).unwrap();
        assert!(result.tiles.len() <= 4);
        assert!(result.statistics.efficiency > 0.0);
    }

    #[test]
    fn test_uniform_partition() {
        let topology = TopologyBuilder::linear(10);
        let partitioner = TopologyPartitioner::new(&topology);
        
        let config = PartitionConfig {
            strategy: PartitionStrategy::Uniform,
            target_partitions: 3,
            min_qubits_per_partition: 2,
            max_qubits_per_partition: 5,
            ..Default::default()
        };

        let result = partitioner.partition(&config).unwrap();
        assert!(!result.tiles.is_empty());
        
        // Check that partition sizes are reasonable
        for tile in &result.tiles {
            assert!(tile.qubit_count() >= 2);
            assert!(tile.qubit_count() <= 5);
        }
    }

    #[test]
    fn test_partition_statistics() {
        let topology = TopologyBuilder::grid(3, 3);
        let partitioner = TopologyPartitioner::new(&topology);
        
        let config = PartitionConfig::default();
        let result = partitioner.partition(&config).unwrap();
        
        assert!(result.statistics.avg_qubits_per_partition > 0.0);
        assert!(result.statistics.efficiency >= 0.0);
        assert!(result.statistics.efficiency <= 1.0);
    }
}