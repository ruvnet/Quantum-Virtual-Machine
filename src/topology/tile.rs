//! Tile-based topology partitioning for circuit scheduling

use crate::{QvmError, Result, Qubit};
use crate::topology::{Topology, Position};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A rectangular tile within the topology
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tile {
    /// Unique tile identifier
    pub id: TileId,
    /// Qubits contained in this tile
    pub qubits: Vec<Qubit>,
    /// Bounding box of the tile
    pub bounds: TileBounds,
    /// Buffer zones around this tile
    pub buffer_zones: HashMap<BufferDirection, usize>,
    /// Tile status
    pub status: TileStatus,
}

/// Tile identifier
pub type TileId = usize;

/// Tile bounding rectangle
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileBounds {
    /// Minimum x coordinate (inclusive)
    pub min_x: i32,
    /// Maximum x coordinate (inclusive)
    pub max_x: i32,
    /// Minimum y coordinate (inclusive)
    pub min_y: i32,
    /// Maximum y coordinate (inclusive)
    pub max_y: i32,
}

/// Buffer zone directions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BufferDirection {
    North,
    South,
    East,
    West,
    Northeast,
    Northwest,
    Southeast,
    Southwest,
}

/// Preferences for tile selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TilePreferences {
    /// Minimum tile width
    pub min_width: u32,
    /// Minimum tile height
    pub min_height: u32,
    /// Maximum tile width
    pub max_width: Option<u32>,
    /// Maximum tile height
    pub max_height: Option<u32>,
    /// Maximum number of qubits
    pub max_qubits: usize,
    /// Required buffer size
    pub buffer_size: usize,
    /// Prefer tiles near center
    pub prefer_center: bool,
    /// Minimum connectivity within tile
    pub min_connectivity: f64,
}

impl Default for TilePreferences {
    fn default() -> Self {
        Self {
            min_width: 2,
            min_height: 2,
            max_width: None,
            max_height: None,
            max_qubits: 10,
            buffer_size: 1,
            prefer_center: true,
            min_connectivity: 0.5,
        }
    }
}

/// Current status of a tile
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TileStatus {
    /// Tile is available for scheduling
    Available,
    /// Tile is currently occupied
    Occupied,
    /// Tile is reserved for future use
    Reserved,
    /// Tile is unavailable due to hardware issues
    Unavailable,
}

impl TileBounds {
    /// Create new tile bounds
    pub fn new(min_x: i32, max_x: i32, min_y: i32, max_y: i32) -> Self {
        Self { min_x, max_x, min_y, max_y }
    }

    /// Get the width of the tile
    pub fn width(&self) -> u32 {
        (self.max_x - self.min_x + 1) as u32
    }

    /// Get the height of the tile
    pub fn height(&self) -> u32 {
        (self.max_y - self.min_y + 1) as u32
    }

    /// Get the area of the tile
    pub fn area(&self) -> u32 {
        self.width() * self.height()
    }

    /// Check if a position is within the tile bounds
    pub fn contains(&self, position: Position) -> bool {
        position.x >= self.min_x 
            && position.x <= self.max_x 
            && position.y >= self.min_y 
            && position.y <= self.max_y
    }

    /// Check if this tile overlaps with another
    pub fn overlaps(&self, other: &TileBounds) -> bool {
        !(self.max_x < other.min_x 
            || self.min_x > other.max_x
            || self.max_y < other.min_y
            || self.min_y > other.max_y)
    }

    /// Get the center position of the tile
    pub fn center(&self) -> Position {
        Position::new(
            (self.min_x + self.max_x) / 2,
            (self.min_y + self.max_y) / 2,
        )
    }

    /// Expand bounds by a margin in all directions
    pub fn expand(&self, margin: u32) -> TileBounds {
        let margin = margin as i32;
        TileBounds::new(
            self.min_x - margin,
            self.max_x + margin,
            self.min_y - margin,
            self.max_y + margin,
        )
    }

    /// Get all positions within the tile bounds
    pub fn positions(&self) -> Vec<Position> {
        let mut positions = Vec::new();
        for x in self.min_x..=self.max_x {
            for y in self.min_y..=self.max_y {
                positions.push(Position::new(x, y));
            }
        }
        positions
    }
}

impl Tile {
    /// Create a new tile
    pub fn new(id: TileId, bounds: TileBounds) -> Self {
        Self {
            id,
            qubits: Vec::new(),
            bounds,
            buffer_zones: HashMap::new(),
            status: TileStatus::Available,
        }
    }

    /// Add a qubit to the tile
    pub fn add_qubit(&mut self, qubit: Qubit) {
        if !self.qubits.contains(&qubit) {
            self.qubits.push(qubit);
        }
    }

    /// Remove a qubit from the tile
    pub fn remove_qubit(&mut self, qubit: Qubit) {
        self.qubits.retain(|&q| q != qubit);
    }

    /// Check if the tile contains a specific qubit
    pub fn contains_qubit(&self, qubit: Qubit) -> bool {
        self.qubits.contains(&qubit)
    }

    /// Get the number of qubits in the tile
    pub fn qubit_count(&self) -> usize {
        self.qubits.len()
    }

    /// Set buffer zone size for a direction
    pub fn set_buffer_zone(&mut self, direction: BufferDirection, size: usize) {
        self.buffer_zones.insert(direction, size);
    }

    /// Get buffer zone size for a direction
    pub fn buffer_zone(&self, direction: BufferDirection) -> usize {
        self.buffer_zones.get(&direction).copied().unwrap_or(0)
    }

    /// Check if the tile is available for scheduling
    pub fn is_available(&self) -> bool {
        matches!(self.status, TileStatus::Available)
    }

    /// Set tile status
    pub fn set_status(&mut self, status: TileStatus) {
        self.status = status;
    }

    /// Get the effective bounds including all buffer zones
    pub fn effective_bounds(&self) -> TileBounds {
        let north_buffer = self.buffer_zone(BufferDirection::North) as i32;
        let south_buffer = self.buffer_zone(BufferDirection::South) as i32;
        let east_buffer = self.buffer_zone(BufferDirection::East) as i32;
        let west_buffer = self.buffer_zone(BufferDirection::West) as i32;

        TileBounds::new(
            self.bounds.min_x - west_buffer,
            self.bounds.max_x + east_buffer,
            self.bounds.min_y - south_buffer,
            self.bounds.max_y + north_buffer,
        )
    }

    /// Check if this tile conflicts with another (considering buffer zones)
    pub fn conflicts_with(&self, other: &Tile) -> bool {
        let self_effective = self.effective_bounds();
        let other_effective = other.effective_bounds();
        self_effective.overlaps(&other_effective)
    }

    /// Calculate the minimum distance to another tile
    pub fn distance_to(&self, other: &Tile) -> f64 {
        let center1 = self.bounds.center();
        let center2 = other.bounds.center();
        center1.euclidean_distance(&center2)
    }
}

/// Tile finder for discovering suitable tiles in a topology
#[derive(Debug, Clone)]
pub struct TileFinder<'a> {
    topology: &'a Topology,
    /// Cache of position to qubit mapping
    position_to_qubit: HashMap<Position, Qubit>,
}

impl<'a> TileFinder<'a> {
    /// Create a new tile finder
    pub fn new(topology: &'a Topology) -> Self {
        let mut position_to_qubit = HashMap::new();
        
        for qubit in topology.qubits() {
            if let Some(position) = topology.position(qubit) {
                position_to_qubit.insert(position, qubit);
            }
        }

        Self {
            topology,
            position_to_qubit,
        }
    }

    /// Find the best tile for a given number of qubits and preferences
    pub fn find_best_tile(&self, required_qubits: usize, preferences: &TilePreferences) -> Result<Option<Tile>> {
        let tiles = self.find_suitable_tiles(required_qubits, preferences)?;
        
        if tiles.is_empty() {
            return Ok(None);
        }

        // Score tiles and return the best one
        let mut best_tile = &tiles[0];
        let mut best_score = self.score_tile(best_tile, preferences);

        for tile in &tiles[1..] {
            let score = self.score_tile(tile, preferences);
            if score > best_score {
                best_score = score;
                best_tile = tile;
            }
        }

        Ok(Some(best_tile.clone()))
    }

    /// Find all tiles that meet the requirements
    pub fn find_suitable_tiles(&self, required_qubits: usize, preferences: &TilePreferences) -> Result<Vec<Tile>> {
        let mut tiles = Vec::new();
        let mut tile_id = 0;

        // Get topology dimensions if available
        let (max_width, max_height) = self.topology.metadata().dimensions.unwrap_or((10, 10));

        // Try different tile sizes starting from minimum required
        let min_width = (required_qubits as f32).sqrt().ceil() as u32;
        let min_height = (required_qubits + min_width as usize - 1) / min_width as usize;

        for width in min_width..=preferences.max_width.unwrap_or(max_width as u32) {
            for height in min_height as u32..=preferences.max_height.unwrap_or(max_height as u32) {
                if (width * height) as usize >= required_qubits {
                    let found_tiles = self.find_rectangular_tile(width, height, 0)?;
                    for tile in found_tiles {
                        if tile.qubits.len() >= required_qubits {
                            tiles.push(tile);
                        }
                    }
                }
            }
        }

        Ok(tiles)
    }

    /// Score a tile based on preferences
    fn score_tile(&self, tile: &Tile, preferences: &TilePreferences) -> f64 {
        let mut score = 0.0;

        // Prefer tiles with more qubits up to max_qubits
        let qubit_count = tile.qubits.len() as f64;
        let max_qubits = preferences.max_qubits as f64;
        if qubit_count <= max_qubits {
            score += qubit_count / max_qubits * 50.0;
        } else {
            score -= (qubit_count - max_qubits) / max_qubits * 20.0;
        }

        // Prefer tiles with good connectivity
        let connectivity = self.calculate_tile_connectivity(tile);
        score += connectivity * 30.0;

        // Prefer compact tiles
        let compactness = self.calculate_tile_compactness(tile);
        score += compactness * 20.0;

        score
    }

    /// Calculate connectivity within a tile
    fn calculate_tile_connectivity(&self, tile: &Tile) -> f64 {
        if tile.qubits.len() < 2 {
            return 1.0;
        }

        let mut connected_pairs = 0;
        let mut total_pairs = 0;

        for i in 0..tile.qubits.len() {
            for j in (i + 1)..tile.qubits.len() {
                total_pairs += 1;
                if self.topology.are_connected(tile.qubits[i], tile.qubits[j]) {
                    connected_pairs += 1;
                }
            }
        }

        if total_pairs == 0 {
            1.0
        } else {
            connected_pairs as f64 / total_pairs as f64
        }
    }

    /// Calculate how compact a tile is
    fn calculate_tile_compactness(&self, tile: &Tile) -> f64 {
        let area = tile.bounds.area() as f64;
        let qubit_count = tile.qubits.len() as f64;
        
        if area == 0.0 {
            0.0
        } else {
            qubit_count / area
        }
    }

    /// Find a rectangular tile that can accommodate the required number of qubits
    pub fn find_rectangular_tile(&self, width: u32, height: u32, buffer_size: usize) -> Result<Vec<Tile>> {
        let mut tiles = Vec::new();
        let mut tile_id = 0;

        // Get topology dimensions if available
        let (max_width, max_height) = self.topology.metadata().dimensions.unwrap_or((10, 10));

        for start_x in 0..=(max_width.saturating_sub(width as usize)) {
            for start_y in 0..=(max_height.saturating_sub(height as usize)) {
                let bounds = TileBounds::new(
                    start_x as i32,
                    (start_x + width as usize - 1) as i32,
                    start_y as i32,
                    (start_y + height as usize - 1) as i32,
                );

                if let Ok(tile) = self.create_tile_from_bounds(tile_id, bounds, buffer_size) {
                    tiles.push(tile);
                    tile_id += 1;
                }
            }
        }

        Ok(tiles)
    }

    /// Find tiles that can fit a specific circuit pattern
    pub fn find_tiles_for_circuit(&self, _required_qubits: usize, _connectivity_pattern: &[(Qubit, Qubit)]) -> Result<Vec<Tile>> {
        // This is a simplified implementation
        // In practice, you'd want to analyze the connectivity pattern more thoroughly
        
        let side_length = (_required_qubits as f64).sqrt().ceil() as u32;
        self.find_rectangular_tile(side_length, side_length, 1)
    }


    /// Create a tile from bounds, populating it with qubits
    pub fn create_tile_from_bounds(&self, tile_id: TileId, bounds: TileBounds, buffer_size: usize) -> Result<Tile> {
        let mut tile = Tile::new(tile_id, bounds.clone());

        // Add qubits that fall within the tile bounds
        for position in bounds.positions() {
            if let Some(&qubit) = self.position_to_qubit.get(&position) {
                tile.add_qubit(qubit);
            }
        }

        // Set uniform buffer zones
        tile.set_buffer_zone(BufferDirection::North, buffer_size);
        tile.set_buffer_zone(BufferDirection::South, buffer_size);
        tile.set_buffer_zone(BufferDirection::East, buffer_size);
        tile.set_buffer_zone(BufferDirection::West, buffer_size);

        // Only return tiles with at least one qubit
        if tile.qubit_count() == 0 {
            return Err(QvmError::topology_error("No qubits found in tile bounds".to_string()));
        }

        Ok(tile)
    }

}


/// Tile manager for tracking tile allocations
#[derive(Debug, Clone)]
pub struct TileManager {
    tiles: HashMap<TileId, Tile>,
    allocated_tiles: HashSet<TileId>,
    next_tile_id: TileId,
}

impl TileManager {
    /// Create a new tile manager
    pub fn new() -> Self {
        Self {
            tiles: HashMap::new(),
            allocated_tiles: HashSet::new(),
            next_tile_id: 0,
        }
    }

    /// Add a tile to the manager
    pub fn add_tile(&mut self, mut tile: Tile) -> TileId {
        let tile_id = self.next_tile_id;
        tile.id = tile_id;
        self.tiles.insert(tile_id, tile);
        self.next_tile_id += 1;
        tile_id
    }

    /// Allocate a tile for use
    pub fn allocate_tile(&mut self, tile_id: TileId) -> Result<()> {
        if !self.tiles.contains_key(&tile_id) {
            return Err(QvmError::allocation_error(
                format!("Tile {} does not exist", tile_id)
            ));
        }

        if self.allocated_tiles.contains(&tile_id) {
            return Err(QvmError::allocation_error(
                format!("Tile {} is already allocated", tile_id)
            ));
        }

        // Check for conflicts with already allocated tiles
        let tile = &self.tiles[&tile_id];
        for &allocated_id in &self.allocated_tiles {
            let allocated_tile = &self.tiles[&allocated_id];
            if tile.conflicts_with(allocated_tile) {
                return Err(QvmError::allocation_error(
                    format!("Tile {} conflicts with allocated tile {}", tile_id, allocated_id)
                ));
            }
        }

        self.allocated_tiles.insert(tile_id);
        
        // Update tile status
        if let Some(tile) = self.tiles.get_mut(&tile_id) {
            tile.set_status(TileStatus::Occupied);
        }

        Ok(())
    }

    /// Release a tile from allocation
    pub fn release_tile(&mut self, tile_id: TileId) -> Result<()> {
        if !self.allocated_tiles.remove(&tile_id) {
            return Err(QvmError::allocation_error(
                format!("Tile {} is not allocated", tile_id)
            ));
        }

        // Update tile status
        if let Some(tile) = self.tiles.get_mut(&tile_id) {
            tile.set_status(TileStatus::Available);
        }

        Ok(())
    }

    /// Get all available tiles
    pub fn available_tiles(&self) -> Vec<&Tile> {
        self.tiles
            .values()
            .filter(|tile| !self.allocated_tiles.contains(&tile.id))
            .collect()
    }

    /// Get an allocated tile
    pub fn get_tile(&self, tile_id: TileId) -> Option<&Tile> {
        self.tiles.get(&tile_id)
    }

    /// Check if a tile is allocated
    pub fn is_allocated(&self, tile_id: TileId) -> bool {
        self.allocated_tiles.contains(&tile_id)
    }
}

impl Default for TileManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;

    #[test]
    fn test_tile_bounds() {
        let bounds = TileBounds::new(0, 2, 0, 2);
        assert_eq!(bounds.width(), 3);
        assert_eq!(bounds.height(), 3);
        assert_eq!(bounds.area(), 9);
        assert!(bounds.contains(Position::new(1, 1)));
        assert!(!bounds.contains(Position::new(3, 1)));
    }

    #[test]
    fn test_tile_creation() {
        let bounds = TileBounds::new(0, 1, 0, 1);
        let mut tile = Tile::new(0, bounds);
        
        tile.add_qubit(Qubit(0));
        tile.add_qubit(Qubit(1));
        
        assert_eq!(tile.qubit_count(), 2);
        assert!(tile.contains_qubit(Qubit(0)));
        assert!(tile.is_available());
    }

    #[test]
    fn test_tile_finder() {
        let topology = TopologyBuilder::grid(4, 4);
        let finder = TileFinder::new(&topology);
        
        let tiles = finder.find_rectangular_tile(2, 2, 1).unwrap();
        assert!(!tiles.is_empty());
        
        // Each 2x2 tile should have 4 qubits
        for tile in &tiles {
            assert!(tile.qubit_count() >= 4);
        }
    }

    #[test]
    fn test_buffer_zones() {
        let bounds = TileBounds::new(1, 2, 1, 2);
        let mut tile = Tile::new(0, bounds);
        tile.set_buffer_zone(BufferDirection::North, 1);
        tile.set_buffer_zone(BufferDirection::South, 1);
        
        let effective = tile.effective_bounds();
        assert_eq!(effective.min_y, 0);
        assert_eq!(effective.max_y, 3);
    }

    #[test]
    fn test_tile_manager() {
        let mut manager = TileManager::new();
        let bounds = TileBounds::new(0, 1, 0, 1);
        let tile = Tile::new(0, bounds);
        
        let tile_id = manager.add_tile(tile);
        manager.allocate_tile(tile_id).unwrap();
        
        assert!(manager.is_allocated(tile_id));
        assert_eq!(manager.available_tiles().len(), 0);
        
        manager.release_tile(tile_id).unwrap();
        assert!(!manager.is_allocated(tile_id));
        assert_eq!(manager.available_tiles().len(), 1);
    }
}