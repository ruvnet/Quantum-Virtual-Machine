//! Topology visualization utilities

use crate::{QvmError, Result, Qubit};
use crate::topology::{Topology, Position, Tile};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Include qubit labels in visualization
    pub show_qubit_labels: bool,
    /// Include connection weights/fidelities
    pub show_connection_weights: bool,
    /// Include positions as coordinates
    pub show_coordinates: bool,
    /// Color scheme for visualization
    pub color_scheme: ColorScheme,
    /// Output format
    pub format: VisualizationFormat,
    /// Scale factor for coordinates
    pub scale_factor: f64,
}

/// Color scheme for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Default black and white
    Default,
    /// Color-coded by connectivity
    Connectivity,
    /// Color-coded by fidelity
    Fidelity,
    /// Color-coded by operational status
    Status,
}

/// Visualization output formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualizationFormat {
    /// ASCII art
    ASCII,
    /// GraphViz DOT format
    GraphViz,
    /// SVG format
    SVG,
    /// JSON coordinates for web visualization
    JSON,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            show_qubit_labels: true,
            show_connection_weights: false,
            show_coordinates: false,
            color_scheme: ColorScheme::Default,
            format: VisualizationFormat::ASCII,
            scale_factor: 1.0,
        }
    }
}

/// Topology visualizer
pub struct TopologyVisualizer {
    config: VisualizationConfig,
}

impl TopologyVisualizer {
    /// Create a new visualizer with default configuration
    pub fn new() -> Self {
        Self {
            config: VisualizationConfig::default(),
        }
    }

    /// Create a visualizer with custom configuration
    pub fn with_config(config: VisualizationConfig) -> Self {
        Self { config }
    }

    /// Generate visualization of the topology
    pub fn visualize(&self, topology: &Topology) -> Result<String> {
        match self.config.format {
            VisualizationFormat::ASCII => self.generate_ascii(topology),
            VisualizationFormat::GraphViz => self.generate_graphviz(topology),
            VisualizationFormat::SVG => self.generate_svg(topology),
            VisualizationFormat::JSON => self.generate_json(topology),
        }
    }

    /// Generate ASCII art representation
    pub fn generate_ascii(&self, topology: &Topology) -> Result<String> {
        let mut output = String::new();
        
        // Add header
        output.push_str(&format!("Topology: {}\n", topology.metadata().name));
        output.push_str(&format!("Qubits: {}, Connections: {}\n", 
                                 topology.qubit_count(), topology.connection_count()));
        output.push_str("=" .repeat(50).as_str());
        output.push('\n');

        // Get positions and bounds
        let qubits = topology.qubits();
        let positions: HashMap<Qubit, Position> = qubits.iter()
            .filter_map(|&q| topology.position(q).map(|p| (q, p)))
            .collect();

        if positions.is_empty() {
            // Generate simple list format if no positions
            return self.generate_ascii_list(topology);
        }

        // Calculate bounds
        let min_x = positions.values().map(|p| p.x).min().unwrap_or(0);
        let max_x = positions.values().map(|p| p.x).max().unwrap_or(0);
        let min_y = positions.values().map(|p| p.y).min().unwrap_or(0);
        let max_y = positions.values().map(|p| p.y).max().unwrap_or(0);

        let width = (max_x - min_x + 1) as usize;
        let height = (max_y - min_y + 1) as usize;

        // Create grid
        let mut grid = vec![vec![' '; width * 3]; height * 2]; // Space for connections

        // Place qubits
        for (qubit, position) in positions {
            let x = ((position.x - min_x) * 3) as usize;
            let y = ((position.y - min_y) * 2) as usize;
            
            if y < grid.len() && x < grid[y].len() {
                if self.config.show_qubit_labels {
                    let label = format!("{}", qubit.index());
                    let chars: Vec<char> = label.chars().collect();
                    for (i, &c) in chars.iter().enumerate() {
                        if x + i < grid[y].len() {
                            grid[y][x + i] = c;
                        }
                    }
                } else {
                    grid[y][x] = '●';
                }
            }
        }

        // Add connections
        for qubit in qubits {
            if let Some(position) = topology.position(qubit) {
                for neighbor in topology.neighbors(qubit) {
                    if let Some(neighbor_pos) = topology.position(neighbor) {
                        self.draw_connection(&mut grid, position, neighbor_pos, min_x, min_y);
                    }
                }
            }
        }

        // Convert grid to string
        for row in grid {
            output.push_str(&row.into_iter().collect::<String>());
            output.push('\n');
        }

        // Add statistics
        output.push('\n');
        output.push_str(&format!("Diameter: {}\n", topology.diameter()));
        output.push_str(&format!("Avg Connectivity: {:.2}\n", topology.connectivity_degree()));

        Ok(output)
    }

    /// Generate ASCII list format for topologies without positions
    fn generate_ascii_list(&self, topology: &Topology) -> Result<String> {
        let mut output = String::new();
        
        for qubit in topology.qubits() {
            let neighbors = topology.neighbors(qubit);
            output.push_str(&format!("Q{}: [", qubit.index()));
            for (i, neighbor) in neighbors.iter().enumerate() {
                if i > 0 { output.push_str(", "); }
                output.push_str(&format!("Q{}", neighbor.index()));
            }
            output.push_str("]\n");
        }

        Ok(output)
    }

    /// Draw connection between two positions in ASCII grid
    fn draw_connection(&self, grid: &mut Vec<Vec<char>>, pos1: Position, pos2: Position, min_x: i32, min_y: i32) {
        let x1 = ((pos1.x - min_x) * 3) as usize;
        let y1 = ((pos1.y - min_y) * 2) as usize;
        let x2 = ((pos2.x - min_x) * 3) as usize;
        let y2 = ((pos2.y - min_y) * 2) as usize;

        // Simple line drawing (horizontal and vertical only)
        if y1 == y2 {
            // Horizontal line
            let start_x = x1.min(x2) + 1;
            let end_x = x1.max(x2);
            for x in start_x..end_x {
                if y1 < grid.len() && x < grid[y1].len() && grid[y1][x] == ' ' {
                    grid[y1][x] = '-';
                }
            }
        } else if x1 == x2 {
            // Vertical line
            let start_y = y1.min(y2) + 1;
            let end_y = y1.max(y2);
            for y in start_y..end_y {
                if y < grid.len() && x1 < grid[y].len() && grid[y][x1] == ' ' {
                    grid[y][x1] = '|';
                }
            }
        }
    }

    /// Generate GraphViz DOT format
    pub fn generate_graphviz(&self, topology: &Topology) -> Result<String> {
        let mut output = String::new();
        
        output.push_str("graph topology {\n");
        output.push_str("    rankdir=LR;\n");
        output.push_str("    node [shape=circle];\n");
        
        // Add nodes
        for qubit in topology.qubits() {
            let label = if self.config.show_qubit_labels {
                format!("Q{}", qubit.index())
            } else {
                qubit.index().to_string()
            };

            let color = match self.config.color_scheme {
                ColorScheme::Connectivity => {
                    let degree = topology.neighbors(qubit).len();
                    match degree {
                        0 => "red",
                        1 => "orange", 
                        2 => "yellow",
                        3 => "lightgreen",
                        4 => "green",
                        _ => "darkgreen",
                    }
                }
                ColorScheme::Fidelity => "lightblue", // Would need fidelity data
                ColorScheme::Status => "lightgray",   // Would need status data
                _ => "lightblue",
            };

            output.push_str(&format!("    {} [label=\"{}\", fillcolor={}, style=filled];\n", 
                                   qubit.index(), label, color));
        }

        output.push('\n');

        // Add edges
        let mut added_edges = HashSet::new();
        for qubit in topology.qubits() {
            for neighbor in topology.neighbors(qubit) {
                let edge = (qubit.index().min(neighbor.index()), qubit.index().max(neighbor.index()));
                if !added_edges.contains(&edge) {
                    added_edges.insert(edge);
                    
                    let edge_label = if self.config.show_connection_weights {
                        " [label=\"1.0\"]" // Placeholder weight
                    } else {
                        ""
                    };

                    output.push_str(&format!("    {} -- {}{};\n", 
                                           qubit.index(), neighbor.index(), edge_label));
                }
            }
        }

        output.push_str("}\n");
        Ok(output)
    }

    /// Generate SVG representation
    pub fn generate_svg(&self, topology: &Topology) -> Result<String> {
        let mut output = String::new();
        
        // Get positions
        let qubits = topology.qubits();
        let positions: HashMap<Qubit, Position> = qubits.iter()
            .filter_map(|&q| topology.position(q).map(|p| (q, p)))
            .collect();

        if positions.is_empty() {
            return Err(QvmError::topology_error("No position information available for SVG generation"));
        }

        // Calculate bounds and scaling
        let min_x = positions.values().map(|p| p.x).min().unwrap_or(0) as f64;
        let max_x = positions.values().map(|p| p.x).max().unwrap_or(0) as f64;
        let min_y = positions.values().map(|p| p.y).min().unwrap_or(0) as f64;
        let max_y = positions.values().map(|p| p.y).max().unwrap_or(0) as f64;

        let margin = 50.0;
        let scale = 50.0 * self.config.scale_factor;
        let width = (max_x - min_x + 1.0) * scale + 2.0 * margin;
        let height = (max_y - min_y + 1.0) * scale + 2.0 * margin;

        // SVG header
        output.push_str(&format!(
            "<svg width=\"{:.0}\" height=\"{:.0}\" xmlns=\"http://www.w3.org/2000/svg\">\n",
            width, height
        ));

        // Background
        output.push_str(&format!(
            "<rect width=\"{:.0}\" height=\"{:.0}\" fill=\"white\" stroke=\"black\" stroke-width=\"1\"/>\n",
            width, height
        ));

        // Draw connections first (so they appear behind qubits)
        let mut added_edges = HashSet::new();
        for qubit in qubits.iter() {
            if let Some(position) = positions.get(qubit) {
                for neighbor in topology.neighbors(*qubit) {
                    if let Some(neighbor_pos) = positions.get(&neighbor) {
                        let edge = (qubit.index().min(neighbor.index()), qubit.index().max(neighbor.index()));
                        if !added_edges.contains(&edge) {
                            added_edges.insert(edge);
                            
                            let x1 = (position.x as f64 - min_x) * scale + margin;
                            let y1 = (position.y as f64 - min_y) * scale + margin;
                            let x2 = (neighbor_pos.x as f64 - min_x) * scale + margin;
                            let y2 = (neighbor_pos.y as f64 - min_y) * scale + margin;

                            output.push_str(&format!(
                                "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"black\" stroke-width=\"2\"/>\n",
                                x1, y1, x2, y2
                            ));
                        }
                    }
                }
            }
        }

        // Draw qubits
        for (qubit, position) in positions {
            let x = (position.x as f64 - min_x) * scale + margin;
            let y = (position.y as f64 - min_y) * scale + margin;

            let fill_color = match self.config.color_scheme {
                ColorScheme::Connectivity => {
                    let degree = topology.neighbors(qubit).len();
                    match degree {
                        0 => "red",
                        1 => "orange",
                        2 => "yellow", 
                        3 => "lightgreen",
                        4 => "green",
                        _ => "darkgreen",
                    }
                }
                _ => "lightblue",
            };

            // Qubit circle
            output.push_str(&format!(
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"15\" fill=\"{}\" stroke=\"black\" stroke-width=\"2\"/>\n",
                x, y, fill_color
            ));

            // Qubit label
            if self.config.show_qubit_labels {
                output.push_str(&format!(
                    "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" dominant-baseline=\"central\" font-family=\"Arial\" font-size=\"12\" fill=\"black\">{}</text>\n",
                    x, y, qubit.index()
                ));
            }

            // Coordinates label
            if self.config.show_coordinates {
                output.push_str(&format!(
                    "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"8\" fill=\"gray\">({},{})</text>\n",
                    x, y + 25.0, position.x, position.y
                ));
            }
        }

        // Title
        output.push_str(&format!(
            "<text x=\"{:.1}\" y=\"20\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"16\" font-weight=\"bold\">{}</text>\n",
            width / 2.0, topology.metadata().name
        ));

        output.push_str("</svg>\n");
        Ok(output)
    }

    /// Generate JSON coordinates for web visualization
    pub fn generate_json(&self, topology: &Topology) -> Result<String> {
        let mut nodes = Vec::new();
        let mut links = Vec::new();

        // Export nodes
        for qubit in topology.qubits() {
            let position = topology.position(qubit);
            let neighbors = topology.neighbors(qubit);
            
            let node = serde_json::json!({
                "id": qubit.index(),
                "label": format!("Q{}", qubit.index()),
                "x": position.map(|p| p.x).unwrap_or(0),
                "y": position.map(|p| p.y).unwrap_or(0),
                "degree": neighbors.len(),
                "neighbors": neighbors.into_iter().map(|q| q.index()).collect::<Vec<_>>()
            });
            nodes.push(node);
        }

        // Export links
        let mut added_edges = HashSet::new();
        for qubit in topology.qubits() {
            for neighbor in topology.neighbors(qubit) {
                let edge = (qubit.index().min(neighbor.index()), qubit.index().max(neighbor.index()));
                if !added_edges.contains(&edge) {
                    added_edges.insert(edge);
                    
                    let link = serde_json::json!({
                        "source": qubit.index(),
                        "target": neighbor.index(),
                        "weight": 1.0
                    });
                    links.push(link);
                }
            }
        }

        let visualization_data = serde_json::json!({
            "topology": {
                "name": topology.metadata().name,
                "type": format!("{:?}", topology.metadata().topology_type),
                "qubit_count": topology.qubit_count(),
                "connection_count": topology.connection_count(),
                "diameter": topology.diameter(),
                "avg_connectivity": topology.connectivity_degree()
            },
            "nodes": nodes,
            "links": links
        });

        serde_json::to_string_pretty(&visualization_data)
            .map_err(|e| QvmError::composition_error(format!("JSON serialization error: {}", e)))
    }

    /// Visualize tiles on the topology
    pub fn visualize_tiles(&self, topology: &Topology, tiles: &[Tile]) -> Result<String> {
        match self.config.format {
            VisualizationFormat::ASCII => self.visualize_tiles_ascii(topology, tiles),
            VisualizationFormat::GraphViz => self.visualize_tiles_graphviz(topology, tiles),
            VisualizationFormat::SVG => self.visualize_tiles_svg(topology, tiles),
            VisualizationFormat::JSON => self.visualize_tiles_json(topology, tiles),
        }
    }

    /// ASCII visualization of tiles
    fn visualize_tiles_ascii(&self, _topology: &Topology, tiles: &[Tile]) -> Result<String> {
        let mut output = String::new();
        
        output.push_str("Tile Visualization\n");
        output.push_str("=".repeat(50).as_str());
        output.push('\n');

        for (_i, tile) in tiles.iter().enumerate() {
            output.push_str(&format!("\nTile {} (Status: {:?}):\n", tile.id, tile.status));
            output.push_str(&format!("  Bounds: ({}, {}) to ({}, {})\n", 
                                   tile.bounds.min_x, tile.bounds.min_y,
                                   tile.bounds.max_x, tile.bounds.max_y));
            output.push_str(&format!("  Qubits: {:?}\n", tile.qubits));
            output.push_str(&format!("  Size: {}x{} ({} total)\n", 
                                   tile.bounds.width(), tile.bounds.height(), tile.bounds.area()));
        }

        Ok(output)
    }

    /// GraphViz visualization of tiles
    fn visualize_tiles_graphviz(&self, topology: &Topology, tiles: &[Tile]) -> Result<String> {
        let mut output = String::new();
        
        output.push_str("graph topology_with_tiles {\n");
        output.push_str("    rankdir=LR;\n");
        
        // Create subgraphs for each tile
        for tile in tiles {
            output.push_str(&format!("    subgraph cluster_{} {{\n", tile.id));
            output.push_str(&format!("        label=\"Tile {}\";\n", tile.id));
            output.push_str("        style=filled;\n");
            output.push_str("        fillcolor=lightgray;\n");
            
            for &qubit in &tile.qubits {
                output.push_str(&format!("        {} [label=\"Q{}\"];\n", 
                                       qubit.index(), qubit.index()));
            }
            
            output.push_str("    }\n");
        }

        // Add connections
        let mut added_edges = HashSet::new();
        for qubit in topology.qubits() {
            for neighbor in topology.neighbors(qubit) {
                let edge = (qubit.index().min(neighbor.index()), qubit.index().max(neighbor.index()));
                if !added_edges.contains(&edge) {
                    added_edges.insert(edge);
                    output.push_str(&format!("    {} -- {};\n", 
                                           qubit.index(), neighbor.index()));
                }
            }
        }

        output.push_str("}\n");
        Ok(output)
    }

    /// SVG visualization of tiles
    fn visualize_tiles_svg(&self, _topology: &Topology, _tiles: &[Tile]) -> Result<String> {
        // Placeholder for SVG tile visualization
        Ok("<svg><!-- Tile visualization not yet implemented --></svg>".to_string())
    }

    /// JSON visualization of tiles
    fn visualize_tiles_json(&self, topology: &Topology, tiles: &[Tile]) -> Result<String> {
        let tile_data: Vec<_> = tiles.iter().map(|tile| {
            serde_json::json!({
                "id": tile.id,
                "qubits": tile.qubits.iter().map(|q| q.index()).collect::<Vec<_>>(),
                "bounds": {
                    "min_x": tile.bounds.min_x,
                    "max_x": tile.bounds.max_x,
                    "min_y": tile.bounds.min_y,
                    "max_y": tile.bounds.max_y,
                    "width": tile.bounds.width(),
                    "height": tile.bounds.height(),
                    "area": tile.bounds.area()
                },
                "status": format!("{:?}", tile.status),
                "buffer_zones": tile.buffer_zones.iter()
                    .map(|(k, v)| (format!("{:?}", k), v))
                    .collect::<HashMap<String, &usize>>()
            })
        }).collect();

        let result = serde_json::json!({
            "topology": topology.metadata().name,
            "tiles": tile_data
        });

        serde_json::to_string_pretty(&result)
            .map_err(|e| QvmError::composition_error(format!("JSON serialization error: {}", e)))
    }
}

impl Default for TopologyVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;
    use crate::topology::{TileBounds, TileStatus};

    #[test]
    fn test_visualizer_creation() {
        let visualizer = TopologyVisualizer::new();
        assert_eq!(visualizer.config.format, VisualizationFormat::ASCII);
    }

    #[test]
    fn test_ascii_visualization() {
        let topology = TopologyBuilder::grid(2, 2);
        let visualizer = TopologyVisualizer::new();
        
        let ascii_output = visualizer.generate_ascii(&topology).unwrap();
        assert!(ascii_output.contains("Topology:"));
        assert!(ascii_output.contains("Qubits: 4"));
    }

    #[test]
    fn test_graphviz_generation() {
        let topology = TopologyBuilder::linear(3);
        let visualizer = TopologyVisualizer::new();
        
        let dot_output = visualizer.generate_graphviz(&topology).unwrap();
        assert!(dot_output.contains("graph topology"));
        assert!(dot_output.contains("0 -- 1"));
        assert!(dot_output.contains("1 -- 2"));
    }

    #[test]
    fn test_json_generation() {
        let topology = TopologyBuilder::linear(3);
        let visualizer = TopologyVisualizer::new();
        
        let json_output = visualizer.generate_json(&topology).unwrap();
        assert!(json_output.contains("nodes"));
        assert!(json_output.contains("links"));
        
        // Should be valid JSON
        let _: serde_json::Value = serde_json::from_str(&json_output).unwrap();
    }

    #[test]
    fn test_svg_generation() {
        let topology = TopologyBuilder::grid(2, 2);
        let visualizer = TopologyVisualizer::new();
        
        let svg_output = visualizer.generate_svg(&topology).unwrap();
        assert!(svg_output.contains("<svg"));
        assert!(svg_output.contains("</svg>"));
        assert!(svg_output.contains("<circle"));
    }

    #[test]
    fn test_tile_visualization() {
        let topology = TopologyBuilder::grid(3, 3);
        let visualizer = TopologyVisualizer::new();
        
        let tile = crate::topology::Tile {
            id: 0,
            qubits: vec![Qubit(0), Qubit(1), Qubit(3), Qubit(4)],
            bounds: TileBounds::new(0, 1, 0, 1),
            buffer_zones: HashMap::new(),
            status: TileStatus::Available,
        };
        
        let tiles = vec![tile];
        let output = visualizer.visualize_tiles(&topology, &tiles).unwrap();
        assert!(output.contains("Tile 0"));
        assert!(output.contains("Bounds:"));
    }
}