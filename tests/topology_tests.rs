//! Comprehensive tests for topology management

use qvm_scheduler::topology::*;
use qvm_scheduler::{Qubit, Result, QvmError};

#[test]
fn test_topology_creation_and_manipulation() {
    let mut topology = Topology::new();
    
    // Add qubits
    assert!(topology.add_qubit(Qubit(0), Some(Position::new(0, 0))).is_ok());
    assert!(topology.add_qubit(Qubit(1), Some(Position::new(1, 0))).is_ok());
    assert!(topology.add_qubit(Qubit(2), Some(Position::new(0, 1))).is_ok());
    
    assert_eq!(topology.qubit_count(), 3);
    
    // Add connections
    let connection = ConnectionEdge {
        fidelity: 0.95,
        distance: 1.0,
        connection_type: ConnectionType::Direct,
        operational: true,
    };
    
    assert!(topology.add_connection(Qubit(0), Qubit(1), connection.clone()).is_ok());
    assert!(topology.add_connection(Qubit(0), Qubit(2), connection.clone()).is_ok());
    
    assert_eq!(topology.connection_count(), 2);
    assert!(topology.are_connected(Qubit(0), Qubit(1)));
    assert!(topology.are_connected(Qubit(0), Qubit(2)));
    assert!(!topology.are_connected(Qubit(1), Qubit(2)));
}

#[test]
fn test_topology_builders() {
    // Test grid topology
    let grid_topology = TopologyBuilder::grid(3, 3);
    assert_eq!(grid_topology.qubit_count(), 9);
    assert_eq!(grid_topology.connection_count(), 12); // 4 * 3 + 4 * 2
    
    // Test linear topology
    let linear_topology = TopologyBuilder::linear(5);
    assert_eq!(linear_topology.qubit_count(), 5);
    assert_eq!(linear_topology.connection_count(), 4);
    
    // Test ring topology
    let ring_topology = TopologyBuilder::ring(4);
    assert_eq!(ring_topology.qubit_count(), 4);
    assert_eq!(ring_topology.connection_count(), 4); // Linear connections + closing connection
    
    // Test star topology
    let star_topology = TopologyBuilder::star(6);
    assert_eq!(star_topology.qubit_count(), 6);
    assert_eq!(star_topology.connection_count(), 5); // Center connected to 5 others
    assert_eq!(star_topology.neighbors(Qubit(0)).len(), 5); // Center has 5 neighbors
}

#[test]
fn test_topology_loaders() {
    let loader = TopologyLoader::new();
    
    // Test JSON serialization/deserialization
    let original_topology = loader.create_linear(5).unwrap();
    let json_str = loader.to_json(&original_topology).unwrap();
    let loaded_topology = loader.from_json(&json_str).unwrap();
    
    assert_eq!(original_topology.qubit_count(), loaded_topology.qubit_count());
    assert_eq!(original_topology.connection_count(), loaded_topology.connection_count());
    assert_eq!(original_topology.metadata().topology_type, loaded_topology.metadata().topology_type);
    
    // Test config format loading
    let config_str = r#"
        name test_topology
        type grid
        qubit 0 0 0
        qubit 1 1 0
        qubit 2 0 1
        qubit 3 1 1
        connection 0 1
        connection 1 3
        connection 3 2
        connection 2 0
    "#;
    
    let config_topology = loader.from_config_string(config_str).unwrap();
    assert_eq!(config_topology.qubit_count(), 4);
    assert_eq!(config_topology.connection_count(), 4);
    assert_eq!(config_topology.metadata().name, "test_topology");
    
    // Test heavy-hex topology
    let heavy_hex = loader.create_heavy_hex(2).unwrap();
    assert!(heavy_hex.qubit_count() > 0);
    assert!(heavy_hex.connection_count() > 0);
}

#[test]
fn test_topology_analysis() {
    let topology = TopologyBuilder::grid(3, 3);
    
    // Test shortest path
    let path = topology.shortest_path(Qubit(0), Qubit(8));
    assert!(path.is_some());
    let path = path.unwrap();
    assert_eq!(path.first(), Some(&Qubit(0)));
    assert_eq!(path.last(), Some(&Qubit(8)));
    
    // Test diameter
    let diameter = topology.diameter();
    assert!(diameter >= 4); // Should be 4 for a 3x3 grid
    
    // Test connectivity degree
    let connectivity = topology.connectivity_degree();
    assert!(connectivity > 0.0);
    assert!(connectivity <= 4.0); // Max 4 neighbors in a grid
    
    // Test qubits within distance
    let nearby = topology.qubits_within_distance(Qubit(4), 1); // Center qubit
    assert_eq!(nearby.len(), 5); // Center + 4 neighbors
    
    // Test connected subgraph
    let subgraph_qubits = vec![Qubit(0), Qubit(1), Qubit(3), Qubit(4)];
    assert!(topology.is_connected_subgraph(&subgraph_qubits));
    
    let disconnected_qubits = vec![Qubit(0), Qubit(8)]; // Corners, not directly connected
    assert!(!topology.is_connected_subgraph(&disconnected_qubits));
}

#[test]
fn test_tile_operations() {
    let bounds = TileBounds::new(0, 2, 0, 2);
    let mut tile = Tile::new(0, bounds.clone());
    
    // Test bounds calculations
    assert_eq!(bounds.width(), 3);
    assert_eq!(bounds.height(), 3);
    assert_eq!(bounds.area(), 9);
    assert!(bounds.contains(Position::new(1, 1)));
    assert!(!bounds.contains(Position::new(3, 1)));
    
    // Test tile operations
    tile.add_qubit(Qubit(0));
    tile.add_qubit(Qubit(1));
    tile.add_qubit(Qubit(2));
    
    assert_eq!(tile.qubit_count(), 3);
    assert!(tile.contains_qubit(Qubit(0)));
    assert!(!tile.contains_qubit(Qubit(3)));
    
    // Test buffer zones
    tile.set_buffer_zone(BufferDirection::North, 2);
    assert_eq!(tile.buffer_zone(BufferDirection::North), 2);
    assert_eq!(tile.buffer_zone(BufferDirection::South), 0);
    
    let effective_bounds = tile.effective_bounds();
    assert_eq!(effective_bounds.max_y, bounds.max_y + 2);
}

#[test]
fn test_tile_finder() {
    let topology = TopologyBuilder::grid(4, 4);
    let finder = TileFinder::new(&topology);
    
    // Test rectangular tile finding
    let tiles = finder.find_rectangular_tile(2, 2, 1).unwrap();
    assert!(!tiles.is_empty());
    
    // Verify tiles have the expected number of qubits
    for tile in &tiles {
        assert!(tile.qubit_count() >= 4); // 2x2 should have 4 qubits
    }
    
    // Test best tile finding
    let preferences = TilePreferences {
        min_width: 2,
        min_height: 2,
        max_qubits: 6,
        buffer_size: 1,
        prefer_center: true,
        min_connectivity: 0.5,
    };
    
    let best_tile = finder.find_best_tile(4, &preferences).unwrap();
    assert!(best_tile.is_some());
}

#[test]
fn test_tile_manager() {
    let mut manager = TileManager::new();
    
    let bounds1 = TileBounds::new(0, 1, 0, 1);
    let bounds2 = TileBounds::new(2, 3, 0, 1);
    let bounds3 = TileBounds::new(0, 1, 2, 3); // Overlaps with bounds1 in buffer zone
    
    let tile1 = Tile::new(0, bounds1);
    let tile2 = Tile::new(1, bounds2);
    let mut tile3 = Tile::new(2, bounds3);
    tile3.set_buffer_zone(BufferDirection::South, 2); // Will conflict with tile1
    
    let tile1_id = manager.add_tile(tile1);
    let tile2_id = manager.add_tile(tile2);
    let tile3_id = manager.add_tile(tile3);
    
    // Test allocation
    assert!(manager.allocate_tile(tile1_id).is_ok());
    assert!(manager.is_allocated(tile1_id));
    
    assert!(manager.allocate_tile(tile2_id).is_ok());
    assert!(manager.is_allocated(tile2_id));
    
    // This should fail due to buffer zone conflict
    assert!(manager.allocate_tile(tile3_id).is_err());
    
    // Test available tiles
    let available = manager.available_tiles();
    assert_eq!(available.len(), 1); // Only tile3 should be available
    
    // Test release
    assert!(manager.release_tile(tile1_id).is_ok());
    assert!(!manager.is_allocated(tile1_id));
    
    let available_after_release = manager.available_tiles();
    assert_eq!(available_after_release.len(), 2); // tile1 and tile3
}

#[test]
fn test_topology_partitioning() {
    let topology = TopologyBuilder::grid(4, 4);
    let partitioner = TopologyPartitioner::new(&topology);
    
    // Test rectangular partitioning
    let config = PartitionConfig {
        strategy: PartitionStrategy::Rectangular,
        target_partitions: 4,
        min_qubits_per_partition: 2,
        max_qubits_per_partition: 8,
        buffer_size: 1,
        balance_factor: 0.8,
    };
    
    let result = partitioner.partition(&config).unwrap();
    assert!(!result.tiles.is_empty());
    assert!(result.tiles.len() <= 4);
    
    // Verify all tiles meet minimum requirements
    for tile in &result.tiles {
        assert!(tile.qubit_count() >= config.min_qubits_per_partition);
        assert!(tile.qubit_count() <= config.max_qubits_per_partition);
    }
    
    // Test statistics
    assert!(result.statistics.avg_qubits_per_partition > 0.0);
    assert!(result.statistics.efficiency >= 0.0);
    assert!(result.statistics.efficiency <= 1.0);
    
    // Test uniform partitioning
    let uniform_config = PartitionConfig {
        strategy: PartitionStrategy::Uniform,
        target_partitions: 3,
        min_qubits_per_partition: 2,
        max_qubits_per_partition: 10,
        ..Default::default()
    };
    
    let uniform_result = partitioner.partition(&uniform_config).unwrap();
    assert!(!uniform_result.tiles.is_empty());
    
    // Test graph-based partitioning
    let graph_config = PartitionConfig {
        strategy: PartitionStrategy::GraphBased,
        target_partitions: 2,
        min_qubits_per_partition: 3,
        max_qubits_per_partition: 12,
        ..Default::default()
    };
    
    let graph_result = partitioner.partition(&graph_config).unwrap();
    assert!(!graph_result.tiles.is_empty());
}

#[test]
fn test_buffer_management() {
    let topology = TopologyBuilder::linear(6);
    let config = BufferConfig {
        default_size: 1,
        min_size: 0,
        max_size: 3,
        decay_factor: 0.8,
        custom_sizes: std::collections::HashMap::new(),
    };
    
    let mut manager = BufferManager::new(config);
    
    // Test buffer size calculation
    let buffer_size = manager.required_buffer_size(Qubit(0), Qubit(1), &topology);
    assert!(buffer_size >= 1);
    
    let distant_buffer_size = manager.required_buffer_size(Qubit(0), Qubit(5), &topology);
    assert!(distant_buffer_size <= buffer_size); // Distant qubits need less buffering
    
    // Test buffer zone creation
    let active_qubits = vec![Qubit(0), Qubit(3)];
    let buffer_zones = manager.create_buffer_zones(&active_qubits, &topology).unwrap();
    
    assert_eq!(buffer_zones.len(), 2);
    assert!(buffer_zones.contains_key(&Qubit(0)));
    assert!(buffer_zones.contains_key(&Qubit(3)));
    
    // Test buffering effectiveness
    let effectiveness_adjacent = manager.buffer_effectiveness(Qubit(0), Qubit(1), &topology);
    let effectiveness_distant = manager.buffer_effectiveness(Qubit(0), Qubit(5), &topology);
    
    assert!(effectiveness_adjacent < effectiveness_distant);
    
    // Test custom buffer sizes
    manager.add_custom_buffer(Qubit(0), Qubit(1), 3);
    let custom_buffer_size = manager.required_buffer_size(Qubit(0), Qubit(1), &topology);
    assert_eq!(custom_buffer_size, 3);
    
    // Test tile validation
    let topology = TopologyBuilder::grid(3, 3);
    let tile1 = Tile {
        id: 0,
        qubits: vec![Qubit(0), Qubit(1)],
        bounds: TileBounds::new(0, 1, 0, 0),
        buffer_zones: std::collections::HashMap::new(),
        status: TileStatus::Available,
    };
    
    let tile2 = Tile {
        id: 1,
        qubits: vec![Qubit(6), Qubit(7)],
        bounds: TileBounds::new(0, 1, 2, 2),
        buffer_zones: std::collections::HashMap::new(),
        status: TileStatus::Available,
    };
    
    let tiles = vec![tile1, tile2];
    let validations = manager.validate_tile_buffers(&tiles, &topology).unwrap();
    
    assert_eq!(validations.len(), 1); // One pair validation
    assert!(validations[0].min_distance > 0);
}

#[test]
fn test_topology_visualization() {
    let topology = TopologyBuilder::grid(3, 3);
    let visualizer = TopologyVisualizer::new();
    
    // Test ASCII visualization
    let ascii_output = visualizer.generate_ascii(&topology).unwrap();
    assert!(ascii_output.contains("Topology:"));
    assert!(ascii_output.contains("Qubits: 9"));
    assert!(ascii_output.contains("Diameter:"));
    
    // Test GraphViz output
    let config = VisualizationConfig {
        format: VisualizationFormat::GraphViz,
        show_qubit_labels: true,
        show_connection_weights: false,
        ..Default::default()
    };
    let graphviz_visualizer = TopologyVisualizer::with_config(config);
    let dot_output = graphviz_visualizer.generate_graphviz(&topology).unwrap();
    
    assert!(dot_output.contains("graph topology"));
    assert!(dot_output.contains("-- "));
    assert!(dot_output.contains("label="));
    
    // Test JSON output
    let json_config = VisualizationConfig {
        format: VisualizationFormat::JSON,
        ..Default::default()
    };
    let json_visualizer = TopologyVisualizer::with_config(json_config);
    let json_output = json_visualizer.generate_json(&topology).unwrap();
    
    assert!(json_output.contains("nodes"));
    assert!(json_output.contains("links"));
    
    // Should be valid JSON
    let _: serde_json::Value = serde_json::from_str(&json_output).unwrap();
    
    // Test SVG output
    let svg_config = VisualizationConfig {
        format: VisualizationFormat::SVG,
        show_coordinates: true,
        ..Default::default()
    };
    let svg_visualizer = TopologyVisualizer::with_config(svg_config);
    let svg_output = svg_visualizer.generate_svg(&topology).unwrap();
    
    assert!(svg_output.contains("<svg"));
    assert!(svg_output.contains("</svg>"));
    assert!(svg_output.contains("<circle"));
    assert!(svg_output.contains("<line"));
    
    // Test tile visualization
    let tile = Tile {
        id: 0,
        qubits: vec![Qubit(0), Qubit(1), Qubit(3), Qubit(4)],
        bounds: TileBounds::new(0, 1, 0, 1),
        buffer_zones: std::collections::HashMap::new(),
        status: TileStatus::Available,
    };
    
    let tiles = vec![tile];
    let tile_output = visualizer.visualize_tiles(&topology, &tiles).unwrap();
    assert!(tile_output.contains("Tile 0"));
    assert!(tile_output.contains("Bounds:"));
}

#[test]
fn test_position_operations() {
    let pos1 = Position::new(0, 0);
    let pos2 = Position::new(3, 4);
    
    // Test distance calculations
    assert_eq!(pos1.manhattan_distance(&pos2), 7);
    assert!((pos1.euclidean_distance(&pos2) - 5.0).abs() < 0.01);
    
    // Test neighbors
    let neighbors = pos1.neighbors();
    assert_eq!(neighbors.len(), 4);
    assert!(neighbors.contains(&Position::new(1, 0)));
    assert!(neighbors.contains(&Position::new(-1, 0)));
    assert!(neighbors.contains(&Position::new(0, 1)));
    assert!(neighbors.contains(&Position::new(0, -1)));
    
    // Test positions within radius
    let nearby = pos1.positions_within_radius(1);
    assert!(nearby.contains(&pos1));
    assert!(nearby.contains(&Position::new(1, 0)));
    assert!(nearby.contains(&Position::new(0, 1)));
    assert!(!nearby.contains(&Position::new(2, 0))); // Outside radius
}

#[test]
fn test_error_handling() {
    let mut topology = Topology::new();
    
    // Test duplicate qubit addition
    topology.add_qubit(Qubit(0), Some(Position::new(0, 0))).unwrap();
    let duplicate_result = topology.add_qubit(Qubit(0), Some(Position::new(1, 0)));
    assert!(duplicate_result.is_err());
    
    // Test connection to non-existent qubit
    let connection = ConnectionEdge {
        fidelity: 0.95,
        distance: 1.0,
        connection_type: ConnectionType::Direct,
        operational: true,
    };
    let invalid_connection = topology.add_connection(Qubit(0), Qubit(1), connection);
    assert!(invalid_connection.is_err());
    
    // Test ring topology with insufficient qubits
    let loader = TopologyLoader::new();
    let small_ring = loader.create_ring(2);
    assert!(small_ring.is_err());
    
    // Test star topology with zero qubits
    let empty_star = loader.create_star(0);
    assert!(empty_star.is_err());
}

#[test]
fn test_topology_metadata() {
    let mut topology = Topology::new();
    
    let mut metadata = TopologyMetadata {
        name: "test_topology".to_string(),
        dimensions: Some((4, 4)),
        topology_type: TopologyType::Grid,
        calibration_time: Some(1234567890),
        properties: std::collections::HashMap::new(),
    };
    
    metadata.properties.insert("vendor".to_string(), "test_vendor".to_string());
    topology.set_metadata(metadata);
    
    let retrieved_metadata = topology.metadata();
    assert_eq!(retrieved_metadata.name, "test_topology");
    assert_eq!(retrieved_metadata.dimensions, Some((4, 4)));
    assert_eq!(retrieved_metadata.topology_type, TopologyType::Grid);
    assert_eq!(retrieved_metadata.calibration_time, Some(1234567890));
    assert_eq!(retrieved_metadata.properties.get("vendor"), Some(&"test_vendor".to_string()));
}