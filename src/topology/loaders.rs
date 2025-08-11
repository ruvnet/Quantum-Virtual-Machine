//! Topology loaders for various formats and standard architectures

use crate::{QvmError, Result, Qubit};
use crate::topology::{Topology, Position, ConnectionEdge, ConnectionType, TopologyMetadata, TopologyType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Topology loader configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderConfig {
    /// Default qubit fidelity
    pub default_fidelity: f64,
    /// Default connection fidelity
    pub default_connection_fidelity: f64,
    /// Default coherence times (T1, T2) in seconds
    pub default_coherence: (f64, f64),
    /// Auto-generate positions for topologies without explicit positions
    pub auto_generate_positions: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            default_fidelity: 0.99,
            default_connection_fidelity: 0.95,
            default_coherence: (100e-6, 50e-6),
            auto_generate_positions: true,
        }
    }
}

/// JSON topology format for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonTopology {
    /// Topology metadata
    pub metadata: TopologyMetadata,
    /// Qubit definitions
    pub qubits: Vec<JsonQubit>,
    /// Connection definitions
    pub connections: Vec<JsonConnection>,
}

/// JSON qubit definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonQubit {
    /// Qubit index
    pub index: usize,
    /// Physical position (optional)
    pub position: Option<(i32, i32)>,
    /// Qubit fidelity
    pub fidelity: Option<f64>,
    /// Coherence times (T1, T2)
    pub coherence: Option<(f64, f64)>,
    /// Operational status
    pub operational: Option<bool>,
}

/// JSON connection definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonConnection {
    /// Source qubit index
    pub source: usize,
    /// Target qubit index
    pub target: usize,
    /// Connection fidelity
    pub fidelity: Option<f64>,
    /// Connection type
    pub connection_type: Option<ConnectionType>,
    /// Operational status
    pub operational: Option<bool>,
}

/// Topology loader for various formats
pub struct TopologyLoader {
    config: LoaderConfig,
}

impl TopologyLoader {
    /// Create a new topology loader with default configuration
    pub fn new() -> Self {
        Self {
            config: LoaderConfig::default(),
        }
    }

    /// Create a topology loader with custom configuration
    pub fn with_config(config: LoaderConfig) -> Self {
        Self { config }
    }

    /// Load topology from JSON string
    pub fn from_json(&self, json_str: &str) -> Result<Topology> {
        let json_topology: JsonTopology = serde_json::from_str(json_str)
            .map_err(|e| QvmError::parse_error(format!("JSON parse error: {}", e), 0))?;

        let mut topology = Topology::new();
        topology.set_metadata(json_topology.metadata);

        // Add qubits
        for json_qubit in json_topology.qubits {
            let qubit = Qubit(json_qubit.index);
            let position = json_qubit.position.map(|(x, y)| Position::new(x, y));
            
            topology.add_qubit(qubit, position)?;

            // Update qubit properties if specified
            if let Some(node_idx) = topology.qubit_to_node.get(&qubit) {
                if let Some(node_weight) = topology.graph.node_weight_mut(*node_idx) {
                    if let Some(fidelity) = json_qubit.fidelity {
                        node_weight.fidelity = fidelity;
                    }
                    if let Some(coherence) = json_qubit.coherence {
                        node_weight.coherence = coherence;
                    }
                    if let Some(operational) = json_qubit.operational {
                        node_weight.operational = operational;
                    }
                }
            }
        }

        // Add connections
        for json_conn in json_topology.connections {
            let qubit1 = Qubit(json_conn.source);
            let qubit2 = Qubit(json_conn.target);
            
            let connection = ConnectionEdge {
                fidelity: json_conn.fidelity.unwrap_or(self.config.default_connection_fidelity),
                distance: self.calculate_distance(&topology, qubit1, qubit2),
                connection_type: json_conn.connection_type.unwrap_or(ConnectionType::Direct),
                operational: json_conn.operational.unwrap_or(true),
            };

            topology.add_connection(qubit1, qubit2, connection)?;
        }

        Ok(topology)
    }

    /// Save topology to JSON string
    pub fn to_json(&self, topology: &Topology) -> Result<String> {
        let mut json_qubits = Vec::new();
        let mut json_connections = Vec::new();

        // Export qubits
        for qubit in topology.qubits() {
            let position = topology.position(qubit).map(|p| (p.x, p.y));
            
            // Get qubit properties from graph
            let (fidelity, coherence, operational) = if let Some(node_idx) = topology.qubit_to_node.get(&qubit) {
                if let Some(node_weight) = topology.graph.node_weight(*node_idx) {
                    (Some(node_weight.fidelity), Some(node_weight.coherence), Some(node_weight.operational))
                } else {
                    (None, None, None)
                }
            } else {
                (None, None, None)
            };

            json_qubits.push(JsonQubit {
                index: qubit.index(),
                position,
                fidelity,
                coherence,
                operational,
            });
        }

        // Export connections
        for qubit1 in topology.qubits() {
            for qubit2 in topology.neighbors(qubit1) {
                // Only export each connection once
                if qubit1.index() < qubit2.index() {
                    // Get connection properties from graph
                    if let (Some(node1), Some(node2)) = (topology.qubit_to_node.get(&qubit1), topology.qubit_to_node.get(&qubit2)) {
                        if let Some(edge_ref) = topology.graph.find_edge(*node1, *node2) {
                            if let Some(edge_weight) = topology.graph.edge_weight(edge_ref) {
                                json_connections.push(JsonConnection {
                                    source: qubit1.index(),
                                    target: qubit2.index(),
                                    fidelity: Some(edge_weight.fidelity),
                                    connection_type: Some(edge_weight.connection_type),
                                    operational: Some(edge_weight.operational),
                                });
                            }
                        }
                    }
                }
            }
        }

        let json_topology = JsonTopology {
            metadata: topology.metadata().clone(),
            qubits: json_qubits,
            connections: json_connections,
        };

        serde_json::to_string_pretty(&json_topology)
            .map_err(|e| QvmError::composition_error(format!("JSON serialization error: {}", e)))
    }

    /// Create a grid topology (2D lattice)
    pub fn create_grid(&self, width: usize, height: usize) -> Result<Topology> {
        let mut topology = Topology::new();
        
        let mut metadata = TopologyMetadata::default();
        metadata.name = format!("grid_{}x{}", width, height);
        metadata.dimensions = Some((width, height));
        metadata.topology_type = TopologyType::Grid;
        topology.set_metadata(metadata);

        // Add qubits
        for y in 0..height {
            for x in 0..width {
                let qubit = Qubit(y * width + x);
                let position = Position::new(x as i32, y as i32);
                topology.add_qubit(qubit, Some(position))?;
            }
        }

        // Add connections (4-connected grid)
        for y in 0..height {
            for x in 0..width {
                let current_qubit = Qubit(y * width + x);
                
                // Right connection
                if x + 1 < width {
                    let right_qubit = Qubit(y * width + x + 1);
                    let connection = ConnectionEdge {
                        fidelity: self.config.default_connection_fidelity,
                        distance: 1.0,
                        connection_type: ConnectionType::Direct,
                        operational: true,
                    };
                    topology.add_connection(current_qubit, right_qubit, connection)?;
                }
                
                // Down connection
                if y + 1 < height {
                    let down_qubit = Qubit((y + 1) * width + x);
                    let connection = ConnectionEdge {
                        fidelity: self.config.default_connection_fidelity,
                        distance: 1.0,
                        connection_type: ConnectionType::Direct,
                        operational: true,
                    };
                    topology.add_connection(current_qubit, down_qubit, connection)?;
                }
            }
        }

        Ok(topology)
    }

    /// Create a linear topology (1D chain)
    pub fn create_linear(&self, size: usize) -> Result<Topology> {
        let mut topology = Topology::new();
        
        let mut metadata = TopologyMetadata::default();
        metadata.name = format!("linear_{}", size);
        metadata.topology_type = TopologyType::Linear;
        topology.set_metadata(metadata);

        // Add qubits
        for i in 0..size {
            let qubit = Qubit(i);
            let position = Position::new(i as i32, 0);
            topology.add_qubit(qubit, Some(position))?;
        }

        // Add connections
        for i in 0..size.saturating_sub(1) {
            let connection = ConnectionEdge {
                fidelity: self.config.default_connection_fidelity,
                distance: 1.0,
                connection_type: ConnectionType::Direct,
                operational: true,
            };
            topology.add_connection(Qubit(i), Qubit(i + 1), connection)?;
        }

        Ok(topology)
    }

    /// Create a heavy-hex topology (IBM quantum computers)
    pub fn create_heavy_hex(&self, distance: usize) -> Result<Topology> {
        let mut topology = Topology::new();
        
        let mut metadata = TopologyMetadata::default();
        metadata.name = format!("heavy_hex_d{}", distance);
        metadata.topology_type = TopologyType::Custom;
        metadata.properties.insert("architecture".to_string(), "heavy_hex".to_string());
        topology.set_metadata(metadata);

        // Heavy-hex topology implementation (simplified)
        // In practice, this would implement the full heavy-hex lattice structure
        let size = distance * 2 + 1;
        let mut qubit_id = 0;
        let mut positions = HashMap::new();

        // Generate hex lattice positions
        for row in 0..size {
            let row_qubits = if row % 2 == 0 { size - row / 2 } else { size - (row + 1) / 2 };
            let offset = row / 2;
            
            for col in 0..row_qubits {
                let qubit = Qubit(qubit_id);
                let x = (col * 2 + offset) as i32;
                let y = (row as f32 * 1.5) as i32;
                let position = Position::new(x, y);
                
                topology.add_qubit(qubit, Some(position))?;
                positions.insert(qubit_id, (x, y));
                qubit_id += 1;
            }
        }

        // Add connections (simplified - would need proper hex lattice connectivity)
        for qubit in topology.qubits() {
            for neighbor in topology.qubits() {
                if qubit.index() != neighbor.index() {
                    if let (Some(pos1), Some(pos2)) = (topology.position(qubit), topology.position(neighbor)) {
                        let distance = pos1.euclidean_distance(&pos2);
                        if distance < 2.5 && distance > 0.5 { // Approximately nearest neighbors
                            let connection = ConnectionEdge {
                                fidelity: self.config.default_connection_fidelity,
                                distance,
                                connection_type: ConnectionType::Direct,
                                operational: true,
                            };
                            topology.add_connection(qubit, neighbor, connection)?;
                        }
                    }
                }
            }
        }

        Ok(topology)
    }

    /// Create a star topology
    pub fn create_star(&self, size: usize) -> Result<Topology> {
        if size == 0 {
            return Err(QvmError::topology_error("Star topology must have at least one qubit"));
        }

        let mut topology = Topology::new();
        
        let mut metadata = TopologyMetadata::default();
        metadata.name = format!("star_{}", size);
        metadata.topology_type = TopologyType::Star;
        topology.set_metadata(metadata);

        // Add qubits
        for i in 0..size {
            let qubit = Qubit(i);
            let position = if i == 0 {
                Position::new(0, 0) // Center
            } else {
                let angle = 2.0 * std::f64::consts::PI * (i - 1) as f64 / (size - 1) as f64;
                Position::new((angle.cos() * 3.0) as i32, (angle.sin() * 3.0) as i32)
            };
            topology.add_qubit(qubit, Some(position))?;
        }

        // Add connections from center to all others
        for i in 1..size {
            let connection = ConnectionEdge {
                fidelity: self.config.default_connection_fidelity,
                distance: 3.0,
                connection_type: ConnectionType::Direct,
                operational: true,
            };
            topology.add_connection(Qubit(0), Qubit(i), connection)?;
        }

        Ok(topology)
    }

    /// Create a ring topology
    pub fn create_ring(&self, size: usize) -> Result<Topology> {
        if size < 3 {
            return Err(QvmError::topology_error("Ring topology must have at least 3 qubits"));
        }

        let mut topology = self.create_linear(size)?;
        topology.metadata.name = format!("ring_{}", size);
        topology.metadata.topology_type = TopologyType::Ring;

        // Add connection from last to first to complete the ring
        let connection = ConnectionEdge {
            fidelity: self.config.default_connection_fidelity,
            distance: 1.0,
            connection_type: ConnectionType::Direct,
            operational: true,
        };
        topology.add_connection(Qubit(size - 1), Qubit(0), connection)?;

        Ok(topology)
    }

    /// Calculate distance between two qubits in topology
    fn calculate_distance(&self, topology: &Topology, qubit1: Qubit, qubit2: Qubit) -> f64 {
        if let (Some(pos1), Some(pos2)) = (topology.position(qubit1), topology.position(qubit2)) {
            pos1.euclidean_distance(&pos2)
        } else {
            1.0 // Default distance if positions not available
        }
    }

    /// Load topology from a configuration file format
    pub fn from_config_string(&self, config_str: &str) -> Result<Topology> {
        // Simple configuration format parser
        let mut topology = Topology::new();
        let mut metadata = TopologyMetadata::default();
        
        for line in config_str.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts.get(0) {
                Some(&"name") => {
                    if let Some(&name) = parts.get(1) {
                        metadata.name = name.to_string();
                    }
                }
                Some(&"type") => {
                    if let Some(&topology_type) = parts.get(1) {
                        metadata.topology_type = match topology_type {
                            "grid" => TopologyType::Grid,
                            "linear" => TopologyType::Linear,
                            "ring" => TopologyType::Ring,
                            "star" => TopologyType::Star,
                            "tree" => TopologyType::Tree,
                            _ => TopologyType::Custom,
                        };
                    }
                }
                Some(&"qubit") => {
                    if let (Some(&index_str), Some(&x_str), Some(&y_str)) = (parts.get(1), parts.get(2), parts.get(3)) {
                        if let (Ok(index), Ok(x), Ok(y)) = (index_str.parse::<usize>(), x_str.parse::<i32>(), y_str.parse::<i32>()) {
                            let qubit = Qubit(index);
                            let position = Position::new(x, y);
                            topology.add_qubit(qubit, Some(position))?;
                        }
                    }
                }
                Some(&"connection") => {
                    if let (Some(&src_str), Some(&tgt_str)) = (parts.get(1), parts.get(2)) {
                        if let (Ok(src), Ok(tgt)) = (src_str.parse::<usize>(), tgt_str.parse::<usize>()) {
                            let connection = ConnectionEdge {
                                fidelity: self.config.default_connection_fidelity,
                                distance: self.calculate_distance(&topology, Qubit(src), Qubit(tgt)),
                                connection_type: ConnectionType::Direct,
                                operational: true,
                            };
                            topology.add_connection(Qubit(src), Qubit(tgt), connection)?;
                        }
                    }
                }
                _ => {
                    // Ignore unknown directives
                }
            }
        }

        topology.set_metadata(metadata);
        Ok(topology)
    }
}

impl Default for TopologyLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loader_creation() {
        let loader = TopologyLoader::new();
        assert_eq!(loader.config.default_fidelity, 0.99);
    }

    #[test]
    fn test_grid_creation() {
        let loader = TopologyLoader::new();
        let topology = loader.create_grid(3, 3).unwrap();
        assert_eq!(topology.qubit_count(), 9);
        assert_eq!(topology.connection_count(), 12); // 4 * 3 + 4 * 2
    }

    #[test]
    fn test_linear_creation() {
        let loader = TopologyLoader::new();
        let topology = loader.create_linear(5).unwrap();
        assert_eq!(topology.qubit_count(), 5);
        assert_eq!(topology.connection_count(), 4);
        assert_eq!(topology.metadata().topology_type, TopologyType::Linear);
    }

    #[test]
    fn test_star_creation() {
        let loader = TopologyLoader::new();
        let topology = loader.create_star(6).unwrap();
        assert_eq!(topology.qubit_count(), 6);
        assert_eq!(topology.connection_count(), 5); // Center connected to 5 others
        assert_eq!(topology.neighbors(Qubit(0)).len(), 5); // Center has 5 neighbors
    }

    #[test]
    fn test_ring_creation() {
        let loader = TopologyLoader::new();
        let topology = loader.create_ring(5).unwrap();
        assert_eq!(topology.qubit_count(), 5);
        assert_eq!(topology.connection_count(), 5); // Linear + one closing connection
        assert_eq!(topology.metadata().topology_type, TopologyType::Ring);
    }

    #[test]
    fn test_json_serialization() {
        let loader = TopologyLoader::new();
        let topology = loader.create_linear(3).unwrap();
        
        let json_str = loader.to_json(&topology).unwrap();
        let deserialized_topology = loader.from_json(&json_str).unwrap();
        
        assert_eq!(topology.qubit_count(), deserialized_topology.qubit_count());
        assert_eq!(topology.connection_count(), deserialized_topology.connection_count());
    }

    #[test]
    fn test_config_format_parsing() {
        let loader = TopologyLoader::new();
        let config_str = r#"
            name test_topology
            type linear
            qubit 0 0 0
            qubit 1 1 0
            qubit 2 2 0
            connection 0 1
            connection 1 2
        "#;
        
        let topology = loader.from_config_string(config_str).unwrap();
        assert_eq!(topology.qubit_count(), 3);
        assert_eq!(topology.connection_count(), 2);
        assert_eq!(topology.metadata().name, "test_topology");
    }
}