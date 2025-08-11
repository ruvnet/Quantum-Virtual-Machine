//! Buffer zone management for crosstalk mitigation

use crate::{QvmError, Result, Qubit};
use crate::topology::{Topology, Position, Tile};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Buffer zone configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferConfig {
    /// Default buffer size (in topology units)
    pub default_size: usize,
    /// Buffer sizes for specific qubit pairs
    pub custom_sizes: HashMap<(Qubit, Qubit), usize>,
    /// Minimum buffer size
    pub min_size: usize,
    /// Maximum buffer size
    pub max_size: usize,
    /// Buffer decay factor (how buffer effectiveness decreases with distance)
    pub decay_factor: f64,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            default_size: 1,
            custom_sizes: HashMap::new(),
            min_size: 0,
            max_size: 5,
            decay_factor: 0.8,
        }
    }
}

/// Buffer zone manager
#[derive(Debug, Clone)]
pub struct BufferManager {
    config: BufferConfig,
    /// Active buffer zones (qubit -> set of buffered qubits)
    active_buffers: HashMap<Qubit, HashSet<Qubit>>,
    /// Buffer effectiveness cache
    effectiveness_cache: HashMap<(Qubit, Qubit), f64>,
}

impl BufferManager {
    /// Create a new buffer manager
    pub fn new(config: BufferConfig) -> Self {
        Self {
            config,
            active_buffers: HashMap::new(),
            effectiveness_cache: HashMap::new(),
        }
    }

    /// Calculate required buffer zone between two qubits
    pub fn required_buffer_size(&self, qubit1: Qubit, qubit2: Qubit, topology: &Topology) -> usize {
        // Check for custom buffer size
        if let Some(&custom_size) = self.config.custom_sizes.get(&(qubit1, qubit2))
            .or_else(|| self.config.custom_sizes.get(&(qubit2, qubit1))) {
            return custom_size;
        }

        // Calculate based on distance and connectivity
        if let Some(path) = topology.shortest_path(qubit1, qubit2) {
            let distance = path.len().saturating_sub(1);
            
            // Closer qubits need larger buffers
            let base_buffer = if distance <= 1 {
                self.config.default_size + 1
            } else if distance <= 2 {
                self.config.default_size
            } else {
                self.config.default_size.saturating_sub(1)
            };

            base_buffer.clamp(self.config.min_size, self.config.max_size)
        } else {
            // No path found, use minimum buffer
            self.config.min_size
        }
    }

    /// Create buffer zones for active circuits
    pub fn create_buffer_zones(
        &mut self,
        active_qubits: &[Qubit],
        topology: &Topology,
    ) -> Result<HashMap<Qubit, HashSet<Qubit>>> {
        let mut buffer_zones = HashMap::new();

        for &qubit in active_qubits {
            let buffer_qubits = self.calculate_buffer_qubits(qubit, active_qubits, topology)?;
            buffer_zones.insert(qubit, buffer_qubits);
        }

        // Store active buffers
        self.active_buffers = buffer_zones.clone();
        Ok(buffer_zones)
    }

    /// Calculate which qubits should be buffered for a given active qubit
    fn calculate_buffer_qubits(
        &self,
        active_qubit: Qubit,
        all_active_qubits: &[Qubit],
        topology: &Topology,
    ) -> Result<HashSet<Qubit>> {
        let mut buffer_qubits = HashSet::new();

        // Get qubit position if available
        let active_position = topology.position(active_qubit);

        // For each other active qubit, determine buffer requirements
        for &other_qubit in all_active_qubits {
            if other_qubit == active_qubit {
                continue;
            }

            let buffer_size = self.required_buffer_size(active_qubit, other_qubit, topology);
            
            // Add qubits within buffer distance
            if let Some(active_pos) = active_position {
                if let Some(other_pos) = topology.position(other_qubit) {
                    // Position-based buffering
                    let buffer_positions = self.get_buffer_positions(active_pos, other_pos, buffer_size);
                    
                    for pos in buffer_positions {
                        // Find qubit at this position
                        for candidate in topology.qubits() {
                            if topology.position(candidate) == Some(pos) {
                                buffer_qubits.insert(candidate);
                            }
                        }
                    }
                } else {
                    // Graph-based buffering
                    let nearby_qubits = topology.qubits_within_distance(active_qubit, buffer_size as u32);
                    for (qubit, _) in nearby_qubits {
                        buffer_qubits.insert(qubit);
                    }
                }
            }
        }

        // Remove the active qubit itself from the buffer zone
        buffer_qubits.remove(&active_qubit);

        Ok(buffer_qubits)
    }

    /// Get positions that should be buffered between two positions
    fn get_buffer_positions(&self, pos1: Position, pos2: Position, buffer_size: usize) -> Vec<Position> {
        let mut positions = Vec::new();

        // Calculate midpoint
        let mid_x = (pos1.x + pos2.x) / 2;
        let mid_y = (pos1.y + pos2.y) / 2;
        let midpoint = Position::new(mid_x, mid_y);

        // Add positions around the midpoint
        for dx in -(buffer_size as i32)..=(buffer_size as i32) {
            for dy in -(buffer_size as i32)..=(buffer_size as i32) {
                let pos = Position::new(midpoint.x + dx, midpoint.y + dy);
                if pos1.manhattan_distance(&pos) <= buffer_size as u32 ||
                   pos2.manhattan_distance(&pos) <= buffer_size as u32 {
                    positions.push(pos);
                }
            }
        }

        positions
    }

    /// Check if two qubits have sufficient buffering
    pub fn is_sufficiently_buffered(&self, qubit1: Qubit, qubit2: Qubit, topology: &Topology) -> bool {
        let required_size = self.required_buffer_size(qubit1, qubit2, topology);
        
        // Check shortest path length
        if let Some(path) = topology.shortest_path(qubit1, qubit2) {
            let actual_distance = path.len().saturating_sub(1);
            actual_distance >= required_size
        } else {
            true // No path means they're naturally isolated
        }
    }

    /// Calculate buffer effectiveness between two qubits
    pub fn buffer_effectiveness(&mut self, qubit1: Qubit, qubit2: Qubit, topology: &Topology) -> f64 {
        // Check cache first
        let key = (qubit1.min(qubit2), qubit1.max(qubit2));
        if let Some(&cached) = self.effectiveness_cache.get(&key) {
            return cached;
        }

        let effectiveness = if let Some(path) = topology.shortest_path(qubit1, qubit2) {
            let distance = path.len().saturating_sub(1);
            let required_buffer = self.required_buffer_size(qubit1, qubit2, topology);
            
            if distance >= required_buffer {
                // Calculate effectiveness based on excess buffer
                let excess = distance.saturating_sub(required_buffer) as f64;
                (1.0 - (-excess * self.config.decay_factor).exp()).min(1.0)
            } else {
                // Insufficient buffer
                let deficit = required_buffer.saturating_sub(distance) as f64;
                (0.1 * (-deficit * 0.5).exp()).max(0.0)
            }
        } else {
            1.0 // Perfect isolation
        };

        // Cache the result
        self.effectiveness_cache.insert(key, effectiveness);
        effectiveness
    }

    /// Get all qubits that should be avoided when scheduling near an active qubit
    pub fn get_avoided_qubits(&self, active_qubit: Qubit) -> HashSet<Qubit> {
        self.active_buffers.get(&active_qubit).cloned().unwrap_or_default()
    }

    /// Clear all active buffer zones
    pub fn clear_buffers(&mut self) {
        self.active_buffers.clear();
        self.effectiveness_cache.clear();
    }

    /// Update buffer configuration
    pub fn update_config(&mut self, config: BufferConfig) {
        self.config = config;
        self.effectiveness_cache.clear(); // Clear cache due to config change
    }

    /// Add custom buffer size for specific qubit pair
    pub fn add_custom_buffer(&mut self, qubit1: Qubit, qubit2: Qubit, buffer_size: usize) {
        self.config.custom_sizes.insert((qubit1, qubit2), buffer_size);
        self.effectiveness_cache.clear();
    }

    /// Validate buffer zones for a set of tiles
    pub fn validate_tile_buffers(&self, tiles: &[Tile], topology: &Topology) -> Result<Vec<BufferValidation>> {
        let mut validations = Vec::new();

        for (i, tile1) in tiles.iter().enumerate() {
            for (j, tile2) in tiles.iter().enumerate().skip(i + 1) {
                let validation = self.validate_tile_pair(tile1, tile2, topology)?;
                validations.push(validation);
            }
        }

        Ok(validations)
    }

    /// Validate buffer between two specific tiles
    fn validate_tile_pair(&self, tile1: &Tile, tile2: &Tile, topology: &Topology) -> Result<BufferValidation> {
        let mut min_distance = u32::MAX;
        let mut problem_pairs = Vec::new();
        let mut total_effectiveness = 0.0;
        let mut pair_count = 0;

        // Check all qubit pairs between tiles
        for &qubit1 in &tile1.qubits {
            for &qubit2 in &tile2.qubits {
                if let Some(path) = topology.shortest_path(qubit1, qubit2) {
                    let distance = path.len().saturating_sub(1) as u32;
                    min_distance = min_distance.min(distance);

                    let required = self.required_buffer_size(qubit1, qubit2, topology);
                    if distance < required as u32 {
                        problem_pairs.push((qubit1, qubit2));
                    }

                    // This would require &mut self, so we skip effectiveness calculation
                    pair_count += 1;
                }
            }
        }

        let avg_effectiveness = if pair_count > 0 { total_effectiveness / pair_count as f64 } else { 1.0 };

        Ok(BufferValidation {
            tile1_id: tile1.id,
            tile2_id: tile2.id,
            min_distance,
            problem_pairs: problem_pairs.clone(),
            avg_effectiveness,
            is_valid: problem_pairs.is_empty(),
        })
    }
}

/// Buffer validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BufferValidation {
    /// First tile ID
    pub tile1_id: usize,
    /// Second tile ID
    pub tile2_id: usize,
    /// Minimum distance between tiles
    pub min_distance: u32,
    /// Qubit pairs with insufficient buffering
    pub problem_pairs: Vec<(Qubit, Qubit)>,
    /// Average buffer effectiveness
    pub avg_effectiveness: f64,
    /// Whether buffering is sufficient
    pub is_valid: bool,
}

/// Buffer zone optimizer
pub struct BufferOptimizer;

impl BufferOptimizer {
    /// Optimize buffer zones to minimize crosstalk while maximizing resource utilization
    pub fn optimize_buffers(
        topology: &Topology,
        active_tiles: &[Tile],
        config: &BufferConfig,
    ) -> Result<BufferConfig> {
        let mut optimized_config = config.clone();

        // Analyze actual crosstalk patterns
        let crosstalk_matrix = Self::analyze_crosstalk_patterns(topology, active_tiles)?;

        // Adjust buffer sizes based on measured crosstalk
        for ((qubit1, qubit2), crosstalk_level) in crosstalk_matrix {
            let current_buffer = optimized_config.custom_sizes.get(&(qubit1, qubit2))
                .copied()
                .unwrap_or(optimized_config.default_size);

            let optimal_buffer = if crosstalk_level > 0.8 {
                (current_buffer + 1).min(optimized_config.max_size)
            } else if crosstalk_level < 0.2 {
                current_buffer.saturating_sub(1).max(optimized_config.min_size)
            } else {
                current_buffer
            };

            if optimal_buffer != optimized_config.default_size {
                optimized_config.custom_sizes.insert((qubit1, qubit2), optimal_buffer);
            }
        }

        Ok(optimized_config)
    }

    /// Analyze crosstalk patterns in the topology
    fn analyze_crosstalk_patterns(
        topology: &Topology,
        tiles: &[Tile],
    ) -> Result<HashMap<(Qubit, Qubit), f64>> {
        let mut crosstalk_matrix = HashMap::new();

        // For each pair of tiles, analyze potential crosstalk
        for tile1 in tiles {
            for tile2 in tiles {
                if tile1.id >= tile2.id {
                    continue;
                }

                for &qubit1 in &tile1.qubits {
                    for &qubit2 in &tile2.qubits {
                        let crosstalk = Self::estimate_crosstalk(qubit1, qubit2, topology);
                        crosstalk_matrix.insert((qubit1, qubit2), crosstalk);
                    }
                }
            }
        }

        Ok(crosstalk_matrix)
    }

    /// Estimate crosstalk level between two qubits (simplified model)
    fn estimate_crosstalk(qubit1: Qubit, qubit2: Qubit, topology: &Topology) -> f64 {
        if let Some(path) = topology.shortest_path(qubit1, qubit2) {
            let distance = path.len().saturating_sub(1) as f64;
            
            // Simple inverse square law for crosstalk
            if distance > 0.0 {
                1.0 / (1.0 + distance * distance)
            } else {
                1.0 // Same qubit
            }
        } else {
            0.0 // No connection, no crosstalk
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;

    #[test]
    fn test_buffer_manager_creation() {
        let config = BufferConfig::default();
        let manager = BufferManager::new(config);
        assert_eq!(manager.config.default_size, 1);
    }

    #[test]
    fn test_required_buffer_size() {
        let topology = TopologyBuilder::grid(3, 3);
        let config = BufferConfig::default();
        let manager = BufferManager::new(config);

        let buffer_size = manager.required_buffer_size(Qubit(0), Qubit(1), &topology);
        assert!(buffer_size >= 1);
    }

    #[test]
    fn test_buffer_zone_creation() {
        let topology = TopologyBuilder::linear(5);
        let config = BufferConfig::default();
        let mut manager = BufferManager::new(config);

        let active_qubits = vec![Qubit(0), Qubit(4)];
        let buffer_zones = manager.create_buffer_zones(&active_qubits, &topology).unwrap();

        assert_eq!(buffer_zones.len(), 2);
        assert!(buffer_zones.contains_key(&Qubit(0)));
        assert!(buffer_zones.contains_key(&Qubit(4)));
    }

    #[test]
    fn test_buffer_effectiveness() {
        let topology = TopologyBuilder::linear(10);
        let config = BufferConfig::default();
        let mut manager = BufferManager::new(config);

        // Adjacent qubits should have low effectiveness
        let effectiveness_adjacent = manager.buffer_effectiveness(Qubit(0), Qubit(1), &topology);
        assert!(effectiveness_adjacent < 0.5);

        // Distant qubits should have high effectiveness
        let effectiveness_distant = manager.buffer_effectiveness(Qubit(0), Qubit(9), &topology);
        assert!(effectiveness_distant > 0.8);
    }

    #[test]
    fn test_custom_buffer_sizes() {
        let topology = TopologyBuilder::grid(3, 3);
        let config = BufferConfig::default();
        let mut manager = BufferManager::new(config);

        manager.add_custom_buffer(Qubit(0), Qubit(1), 3);
        let buffer_size = manager.required_buffer_size(Qubit(0), Qubit(1), &topology);
        assert_eq!(buffer_size, 3);
    }
}