//! Hardware topology abstraction and management
//!
//! Provides graph-based representation of quantum hardware topologies,
//! tile-based partitioning, and buffer zone management for crosstalk mitigation.

pub mod tile;
pub mod partition;
pub mod buffer;
pub mod loaders;
pub mod visualization;

pub use tile::*;
pub use partition::*;
pub use buffer::*;
pub use loaders::*;
pub use visualization::*;

use crate::{QvmError, Result, Qubit};
use petgraph::Graph;
use petgraph::graph::{NodeIndex, UnGraph};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// 2D position in the topology
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    pub x: i32,
    pub y: i32,
}

impl Position {
    /// Create a new position
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Calculate Manhattan distance to another position
    pub fn manhattan_distance(&self, other: &Position) -> u32 {
        ((self.x - other.x).abs() + (self.y - other.y).abs()) as u32
    }

    /// Calculate Euclidean distance to another position
    pub fn euclidean_distance(&self, other: &Position) -> f64 {
        let dx = (self.x - other.x) as f64;
        let dy = (self.y - other.y) as f64;
        (dx * dx + dy * dy).sqrt()
    }

    /// Get neighboring positions (4-connected)
    pub fn neighbors(&self) -> [Position; 4] {
        [
            Position::new(self.x + 1, self.y),
            Position::new(self.x - 1, self.y),
            Position::new(self.x, self.y + 1),
            Position::new(self.x, self.y - 1),
        ]
    }

    /// Get all positions within a given radius
    pub fn positions_within_radius(&self, radius: u32) -> Vec<Position> {
        let mut positions = Vec::new();
        let r = radius as i32;
        
        for dx in -r..=r {
            for dy in -r..=r {
                let pos = Position::new(self.x + dx, self.y + dy);
                if self.manhattan_distance(&pos) <= radius {
                    positions.push(pos);
                }
            }
        }
        
        positions
    }
}

/// Quantum hardware topology representation
#[derive(Debug, Clone)]
pub struct Topology {
    /// Graph representation of qubit connectivity
    graph: UnGraph<QubitNode, ConnectionEdge>,
    /// Mapping from qubit index to graph node
    qubit_to_node: HashMap<Qubit, NodeIndex>,
    /// Mapping from graph node to qubit index
    node_to_qubit: HashMap<NodeIndex, Qubit>,
    /// Physical positions of qubits (if available)
    positions: HashMap<Qubit, Position>,
    /// Topology metadata
    metadata: TopologyMetadata,
}

/// Qubit node properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QubitNode {
    /// Qubit identifier
    pub qubit: Qubit,
    /// Physical position (optional)
    pub position: Option<Position>,
    /// Quality metrics
    pub fidelity: f64,
    /// Coherence times (T1, T2)
    pub coherence: (f64, f64),
    /// Is this qubit operational?
    pub operational: bool,
}

/// Connection edge properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionEdge {
    /// Connection strength/fidelity
    pub fidelity: f64,
    /// Physical distance
    pub distance: f64,
    /// Connection type
    pub connection_type: ConnectionType,
    /// Is this connection operational?
    pub operational: bool,
}

/// Type of qubit connection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// Direct physical coupling
    Direct,
    /// Resonant coupling
    Resonant,
    /// Optical connection
    Optical,
    /// Virtual/routing connection
    Virtual,
}

/// Topology metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TopologyMetadata {
    /// Topology name
    pub name: String,
    /// Physical dimensions
    pub dimensions: Option<(usize, usize)>,
    /// Topology type classification
    pub topology_type: TopologyType,
    /// Calibration timestamp
    pub calibration_time: Option<u64>,
    /// Custom properties
    pub properties: HashMap<String, String>,
}

/// Classification of topology types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TopologyType {
    #[default]
    /// Grid/lattice topology
    Grid,
    /// Linear chain
    Linear,
    /// Ring topology
    Ring,
    /// Star topology
    Star,
    /// Fully connected
    Complete,
    /// Tree topology
    Tree,
    /// Custom/irregular topology
    Custom,
}

impl Topology {
    /// Create a new empty topology
    pub fn new() -> Self {
        Self {
            graph: Graph::new_undirected(),
            qubit_to_node: HashMap::new(),
            node_to_qubit: HashMap::new(),
            positions: HashMap::new(),
            metadata: TopologyMetadata::default(),
        }
    }

    /// Add a qubit to the topology
    pub fn add_qubit(&mut self, qubit: Qubit, position: Option<Position>) -> Result<()> {
        if self.qubit_to_node.contains_key(&qubit) {
            return Err(QvmError::topology_error(
                format!("Qubit {:?} already exists in topology", qubit)
            ));
        }

        let node_data = QubitNode {
            qubit,
            position,
            fidelity: 0.99, // Default fidelity
            coherence: (100e-6, 50e-6), // Default T1, T2 in seconds
            operational: true,
        };

        let node_idx = self.graph.add_node(node_data);
        self.qubit_to_node.insert(qubit, node_idx);
        self.node_to_qubit.insert(node_idx, qubit);

        if let Some(pos) = position {
            self.positions.insert(qubit, pos);
        }

        Ok(())
    }

    /// Add a connection between two qubits
    pub fn add_connection(&mut self, qubit1: Qubit, qubit2: Qubit, connection: ConnectionEdge) -> Result<()> {
        let node1 = self.qubit_to_node.get(&qubit1)
            .ok_or_else(|| QvmError::topology_error(format!("Qubit {:?} not found", qubit1)))?;
        let node2 = self.qubit_to_node.get(&qubit2)
            .ok_or_else(|| QvmError::topology_error(format!("Qubit {:?} not found", qubit2)))?;

        self.graph.add_edge(*node1, *node2, connection);
        Ok(())
    }

    /// Get the number of qubits in the topology
    pub fn qubit_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the number of connections in the topology
    pub fn connection_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Check if two qubits are connected
    pub fn are_connected(&self, qubit1: Qubit, qubit2: Qubit) -> bool {
        if let (Some(&node1), Some(&node2)) = (self.qubit_to_node.get(&qubit1), self.qubit_to_node.get(&qubit2)) {
            self.graph.find_edge(node1, node2).is_some()
        } else {
            false
        }
    }

    /// Get all qubits connected to a given qubit
    pub fn neighbors(&self, qubit: Qubit) -> Vec<Qubit> {
        if let Some(&node) = self.qubit_to_node.get(&qubit) {
            self.graph
                .neighbors(node)
                .filter_map(|neighbor_node| self.node_to_qubit.get(&neighbor_node))
                .copied()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Calculate shortest path between two qubits
    pub fn shortest_path(&self, start: Qubit, end: Qubit) -> Option<Vec<Qubit>> {
        let start_node = self.qubit_to_node.get(&start)?;
        let end_node = self.qubit_to_node.get(&end)?;

        // BFS to find shortest path
        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();

        queue.push_back(*start_node);
        visited.insert(*start_node);

        while let Some(current) = queue.pop_front() {
            if current == *end_node {
                // Reconstruct path
                let mut path = Vec::new();
                let mut node = current;
                
                loop {
                    if let Some(&qubit) = self.node_to_qubit.get(&node) {
                        path.push(qubit);
                    }
                    
                    if let Some(&parent_node) = parent.get(&node) {
                        node = parent_node;
                    } else {
                        break;
                    }
                }
                
                path.reverse();
                return Some(path);
            }

            for neighbor in self.graph.neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);
                }
            }
        }

        None
    }

    /// Get the diameter of the topology (longest shortest path)
    pub fn diameter(&self) -> u32 {
        let mut max_distance = 0;
        let qubits: Vec<_> = self.qubit_to_node.keys().copied().collect();

        for i in 0..qubits.len() {
            for j in (i + 1)..qubits.len() {
                if let Some(path) = self.shortest_path(qubits[i], qubits[j]) {
                    max_distance = max_distance.max(path.len() as u32 - 1);
                }
            }
        }

        max_distance
    }

    /// Calculate connectivity degree (average number of neighbors)
    pub fn connectivity_degree(&self) -> f64 {
        if self.qubit_count() == 0 {
            return 0.0;
        }

        let total_degree: usize = self.qubit_to_node
            .keys()
            .map(|&qubit| self.neighbors(qubit).len())
            .sum();

        total_degree as f64 / self.qubit_count() as f64
    }

    /// Get all qubits in the topology
    pub fn qubits(&self) -> Vec<Qubit> {
        self.qubit_to_node.keys().copied().collect()
    }

    /// Get qubit position
    pub fn position(&self, qubit: Qubit) -> Option<Position> {
        self.positions.get(&qubit).copied()
    }

    /// Set qubit position
    pub fn set_position(&mut self, qubit: Qubit, position: Position) {
        self.positions.insert(qubit, position);
        
        // Update node data if qubit exists
        if let Some(&node_idx) = self.qubit_to_node.get(&qubit) {
            if let Some(node_weight) = self.graph.node_weight_mut(node_idx) {
                node_weight.position = Some(position);
            }
        }
    }

    /// Get topology metadata
    pub fn metadata(&self) -> &TopologyMetadata {
        &self.metadata
    }

    /// Set topology metadata
    pub fn set_metadata(&mut self, metadata: TopologyMetadata) {
        self.metadata = metadata;
    }

    /// Find all qubits within a given distance from a center qubit
    pub fn qubits_within_distance(&self, center: Qubit, max_distance: u32) -> Vec<(Qubit, u32)> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back((center, 0));
        visited.insert(center);

        while let Some((current_qubit, distance)) = queue.pop_front() {
            result.push((current_qubit, distance));

            if distance < max_distance {
                for neighbor in self.neighbors(current_qubit) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back((neighbor, distance + 1));
                    }
                }
            }
        }

        result.sort_by_key(|&(_, dist)| dist);
        result
    }

    /// Check if a set of qubits forms a connected subgraph
    pub fn is_connected_subgraph(&self, qubits: &[Qubit]) -> bool {
        if qubits.is_empty() {
            return true;
        }

        let qubit_set: HashSet<_> = qubits.iter().copied().collect();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start from first qubit
        queue.push_back(qubits[0]);
        visited.insert(qubits[0]);

        while let Some(current) = queue.pop_front() {
            for neighbor in self.neighbors(current) {
                if qubit_set.contains(&neighbor) && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        visited.len() == qubits.len()
    }

    /// Find the minimum spanning subgraph for a set of qubits
    pub fn minimum_spanning_subgraph(&self, qubits: &[Qubit]) -> Vec<(Qubit, Qubit)> {
        // Implementation of minimum spanning tree for the subgraph
        // This is a simplified version - could be improved with proper MST algorithms
        let mut edges = Vec::new();
        let qubit_set: HashSet<_> = qubits.iter().copied().collect();

        for &qubit1 in qubits {
            for &qubit2 in qubits {
                if qubit1 < qubit2 && self.are_connected(qubit1, qubit2) {
                    edges.push((qubit1, qubit2));
                }
            }
        }

        // For now, return all valid edges (could implement proper MST algorithm)
        edges
    }
}

impl Default for Topology {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating common topology patterns
pub struct TopologyBuilder;

impl TopologyBuilder {
    /// Create a grid topology
    pub fn grid(width: usize, height: usize) -> Topology {
        TopologyLoader::new().create_grid(width, height).unwrap_or_else(|_| Topology::new())
    }

    /// Create a grid topology (legacy method for backwards compatibility)
    pub fn grid_detailed(width: usize, height: usize) -> Topology {
        let mut topology = Topology::new();
        topology.metadata.name = format!("grid_{}x{}", width, height);
        topology.metadata.dimensions = Some((width, height));
        topology.metadata.topology_type = TopologyType::Grid;

        // Add qubits
        for y in 0..height {
            for x in 0..width {
                let qubit = Qubit(y * width + x);
                let position = Position::new(x as i32, y as i32);
                topology.add_qubit(qubit, Some(position)).unwrap();
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
                        fidelity: 0.95,
                        distance: 1.0,
                        connection_type: ConnectionType::Direct,
                        operational: true,
                    };
                    topology.add_connection(current_qubit, right_qubit, connection).unwrap();
                }
                
                // Down connection
                if y + 1 < height {
                    let down_qubit = Qubit((y + 1) * width + x);
                    let connection = ConnectionEdge {
                        fidelity: 0.95,
                        distance: 1.0,
                        connection_type: ConnectionType::Direct,
                        operational: true,
                    };
                    topology.add_connection(current_qubit, down_qubit, connection).unwrap();
                }
            }
        }

        topology
    }

    /// Create a linear topology
    pub fn linear(size: usize) -> Topology {
        TopologyLoader::new().create_linear(size).unwrap_or_else(|_| Topology::new())
    }

    /// Create a linear topology (legacy method for backwards compatibility)
    pub fn linear_detailed(size: usize) -> Topology {
        let mut topology = Topology::new();
        topology.metadata.name = format!("linear_{}", size);
        topology.metadata.topology_type = TopologyType::Linear;

        // Add qubits
        for i in 0..size {
            let qubit = Qubit(i);
            let position = Position::new(i as i32, 0);
            topology.add_qubit(qubit, Some(position)).unwrap();
        }

        // Add connections
        for i in 0..size.saturating_sub(1) {
            let connection = ConnectionEdge {
                fidelity: 0.95,
                distance: 1.0,
                connection_type: ConnectionType::Direct,
                operational: true,
            };
            topology.add_connection(Qubit(i), Qubit(i + 1), connection).unwrap();
        }

        topology
    }

    /// Create a ring topology
    pub fn ring(size: usize) -> Topology {
        TopologyLoader::new().create_ring(size).unwrap_or_else(|_| Topology::new())
    }

    /// Create a ring topology (legacy method for backwards compatibility)
    pub fn ring_detailed(size: usize) -> Topology {
        let mut topology = Self::linear(size);
        topology.metadata.name = format!("ring_{}", size);
        topology.metadata.topology_type = TopologyType::Ring;

        // Add connection from last to first
        if size > 2 {
            let connection = ConnectionEdge {
                fidelity: 0.95,
                distance: 1.0,
                connection_type: ConnectionType::Direct,
                operational: true,
            };
            topology.add_connection(Qubit(size - 1), Qubit(0), connection).unwrap();
        }

        topology
    }

    /// Create a star topology
    pub fn star(size: usize) -> Topology {
        TopologyLoader::new().create_star(size).unwrap_or_else(|_| Topology::new())
    }

    /// Create a star topology (legacy method for backwards compatibility)
    pub fn star_detailed(size: usize) -> Topology {
        let mut topology = Topology::new();
        topology.metadata.name = format!("star_{}", size);
        topology.metadata.topology_type = TopologyType::Star;

        // Add qubits
        for i in 0..size {
            let qubit = Qubit(i);
            let position = if i == 0 {
                Position::new(0, 0) // Center
            } else {
                let angle = 2.0 * std::f64::consts::PI * (i - 1) as f64 / (size - 1) as f64;
                Position::new((angle.cos() * 2.0) as i32, (angle.sin() * 2.0) as i32)
            };
            topology.add_qubit(qubit, Some(position)).unwrap();
        }

        // Add connections from center to all others
        for i in 1..size {
            let connection = ConnectionEdge {
                fidelity: 0.95,
                distance: 2.0,
                connection_type: ConnectionType::Direct,
                operational: true,
            };
            topology.add_connection(Qubit(0), Qubit(i), connection).unwrap();
        }

        topology
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_creation() {
        let mut topology = Topology::new();
        topology.add_qubit(Qubit(0), Some(Position::new(0, 0))).unwrap();
        topology.add_qubit(Qubit(1), Some(Position::new(1, 0))).unwrap();

        let connection = ConnectionEdge {
            fidelity: 0.95,
            distance: 1.0,
            connection_type: ConnectionType::Direct,
            operational: true,
        };
        topology.add_connection(Qubit(0), Qubit(1), connection).unwrap();

        assert_eq!(topology.qubit_count(), 2);
        assert_eq!(topology.connection_count(), 1);
        assert!(topology.are_connected(Qubit(0), Qubit(1)));
    }

    #[test]
    fn test_grid_topology() {
        let topology = TopologyBuilder::grid(3, 3);
        assert_eq!(topology.qubit_count(), 9);
        assert_eq!(topology.connection_count(), 12); // 4 * 3 + 4 * 2

        // Check center qubit has 4 neighbors
        assert_eq!(topology.neighbors(Qubit(4)).len(), 4);
        
        // Check corner qubit has 2 neighbors
        assert_eq!(topology.neighbors(Qubit(0)).len(), 2);
    }

    #[test]
    fn test_shortest_path() {
        let topology = TopologyBuilder::linear(5);
        let path = topology.shortest_path(Qubit(0), Qubit(4)).unwrap();
        assert_eq!(path, vec![Qubit(0), Qubit(1), Qubit(2), Qubit(3), Qubit(4)]);
    }

    #[test]
    fn test_connectivity_degree() {
        let topology = TopologyBuilder::linear(5);
        let degree = topology.connectivity_degree();
        assert!((degree - 1.6).abs() < 0.1); // Most have 2 neighbors, ends have 1
    }

    #[test]
    fn test_position_distance() {
        let pos1 = Position::new(0, 0);
        let pos2 = Position::new(3, 4);
        
        assert_eq!(pos1.manhattan_distance(&pos2), 7);
        assert!((pos1.euclidean_distance(&pos2) - 5.0).abs() < 0.01);
    }
}