//! Circuit composition and output generation

pub mod mapping;
pub mod output;
pub mod timing;

pub use mapping::*;
pub use output::*;
pub use timing::*;

use crate::{QvmError, Result, QuantumCircuit, Topology, Qubit};
use crate::scheduler::{Schedule, Assignment};
use serde::{Deserialize, Serialize};

/// Composite circuit containing multiple scheduled circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeCircuit {
    /// Individual circuit components
    circuits: Vec<ComposedCircuit>,
    /// Global metadata
    metadata: CompositeMetadata,
    /// Total execution time
    total_duration: u64,
    /// Resource usage summary
    resource_summary: ResourceSummary,
}

/// Individual composed circuit within the composite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposedCircuit {
    /// Original job ID
    pub job_id: usize,
    /// Modified circuit with mapping applied
    pub circuit: QuantumCircuit,
    /// Timing information
    pub timing: CircuitTiming,
    /// Resource mapping
    pub mapping: ResourceMapping,
}

/// Metadata for the composite circuit
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompositeMetadata {
    /// Creation timestamp
    pub created_at: u64,
    /// Total number of circuits
    pub circuit_count: usize,
    /// Scheduling algorithm used
    pub scheduler_algorithm: String,
    /// Composition strategy
    pub composition_strategy: String,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Resource usage summary
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceSummary {
    /// Total qubits used
    pub total_qubits_used: usize,
    /// Peak concurrent qubit usage
    pub peak_qubit_usage: usize,
    /// Total classical bits used
    pub total_classical_used: usize,
    /// Resource utilization efficiency
    pub utilization_efficiency: f64,
}

/// Quality metrics for the composition
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Expected overall fidelity
    pub expected_fidelity: f64,
    /// Estimated crosstalk effects
    pub crosstalk_estimate: f64,
    /// Timing precision
    pub timing_precision: f64,
    /// Composition optimality score
    pub optimality_score: f64,
}

impl CompositeCircuit {
    /// Create a new empty composite circuit
    pub fn new() -> Self {
        Self {
            circuits: Vec::new(),
            metadata: CompositeMetadata::default(),
            total_duration: 0,
            resource_summary: ResourceSummary::default(),
        }
    }

    /// Add a composed circuit
    pub fn add_circuit(&mut self, circuit: ComposedCircuit) {
        self.total_duration = self.total_duration.max(
            circuit.timing.start_time + circuit.timing.duration
        );
        self.circuits.push(circuit);
        self.update_metadata();
    }

    /// Get all circuits
    pub fn circuits(&self) -> &[ComposedCircuit] {
        &self.circuits
    }

    /// Get circuit count
    pub fn circuit_count(&self) -> usize {
        self.circuits.len()
    }

    /// Get total duration
    pub fn total_duration(&self) -> u64 {
        self.total_duration
    }

    /// Get total number of qubits used
    pub fn total_qubits(&self) -> usize {
        self.resource_summary.total_qubits_used
    }

    /// Get total number of classical bits used
    pub fn total_cbits(&self) -> usize {
        self.resource_summary.total_classical_used
    }

    /// Convert to OpenQASM 3 output
    pub fn to_qasm(&self) -> Result<String> {
        let output_generator = OutputGenerator::new();
        output_generator.generate_qasm(self)
    }

    /// Update metadata based on current circuits
    fn update_metadata(&mut self) {
        self.metadata.circuit_count = self.circuits.len();
        
        // Update resource summary
        let mut used_qubits = std::collections::HashSet::new();
        let mut used_classical = std::collections::HashSet::new();
        
        for circuit in &self.circuits {
            for &qubit in &circuit.mapping.qubit_mapping {
                used_qubits.insert(qubit);
            }
            for &classical in &circuit.mapping.classical_mapping {
                used_classical.insert(classical);
            }
        }
        
        self.resource_summary.total_qubits_used = used_qubits.len();
        self.resource_summary.total_classical_used = used_classical.len();
        
        // Calculate peak usage (simplified)
        self.resource_summary.peak_qubit_usage = used_qubits.len();
        
        // Update quality metrics
        self.update_quality_metrics();
    }

    /// Update quality metrics
    fn update_quality_metrics(&mut self) {
        if self.circuits.is_empty() {
            return;
        }

        // Calculate expected fidelity (simplified)
        let avg_fidelity: f64 = self.circuits.iter()
            .map(|c| c.circuit.operations.len() as f64 * 0.99) // Simplified
            .sum::<f64>() / self.circuits.len() as f64;
        
        self.metadata.quality_metrics.expected_fidelity = avg_fidelity.min(1.0);
        
        // Timing precision based on duration spread
        let durations: Vec<u64> = self.circuits.iter()
            .map(|c| c.timing.duration)
            .collect();
        
        if let (Some(&min_dur), Some(&max_dur)) = (durations.iter().min(), durations.iter().max()) {
            self.metadata.quality_metrics.timing_precision = if max_dur > 0 {
                1.0 - (max_dur - min_dur) as f64 / max_dur as f64
            } else {
                1.0
            };
        }
        
        // Optimality score (placeholder)
        self.metadata.quality_metrics.optimality_score = 0.8;
    }
}

impl Default for CompositeCircuit {
    fn default() -> Self {
        Self::new()
    }
}

/// Main circuit composer
#[derive(Debug, Clone)]
pub struct CircuitComposer {
    topology: Topology,
    mapper: QubitMapper,
    timer: CircuitTimer,
    config: ComposerConfig,
}

/// Composer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposerConfig {
    /// Enable timing optimization
    pub optimize_timing: bool,
    /// Enable qubit mapping optimization
    pub optimize_mapping: bool,
    /// Buffer insertion strategy
    pub buffer_strategy: BufferStrategy,
    /// Output format preferences
    pub output_format: OutputFormat,
    /// Quality optimization level (0-3)
    pub optimization_level: u8,
}

impl Default for ComposerConfig {
    fn default() -> Self {
        Self {
            optimize_timing: true,
            optimize_mapping: true,
            buffer_strategy: BufferStrategy::Automatic,
            output_format: OutputFormat::OpenQASM3,
            optimization_level: 2,
        }
    }
}

/// Buffer insertion strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BufferStrategy {
    /// No buffer insertion
    None,
    /// Automatic buffer insertion based on topology
    Automatic,
    /// Manual buffer zones
    Manual,
    /// Adaptive based on circuit analysis
    Adaptive,
}

/// Output format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// OpenQASM 3.0
    OpenQASM3,
    /// OpenQASM 2.0
    OpenQASM2,
    /// Custom format
    Custom,
}

impl CircuitComposer {
    /// Create a new circuit composer
    pub fn new(topology: Topology) -> Self {
        Self {
            mapper: QubitMapper::new(&topology),
            timer: CircuitTimer::new(),
            topology,
            config: ComposerConfig::default(),
        }
    }

    /// Create composer with custom configuration
    pub fn with_config(topology: Topology, config: ComposerConfig) -> Self {
        Self {
            mapper: QubitMapper::new(&topology),
            timer: CircuitTimer::new(),
            topology,
            config,
        }
    }

    /// Compose a complete schedule into a composite circuit
    pub async fn compose(&mut self, schedule: Schedule) -> Result<CompositeCircuit> {
        let mut composite = CompositeCircuit::new();
        composite.metadata.scheduler_algorithm = schedule.metadata.algorithm.clone();
        composite.metadata.composition_strategy = "Sequential".to_string();

        // Process each assignment
        for assignment in schedule.assignments {
            let composed_circuit = self.compose_assignment(assignment).await?;
            composite.add_circuit(composed_circuit);
        }

        // Apply optimizations if enabled
        if self.config.optimization_level > 0 {
            self.optimize_composite(&mut composite).await?;
        }

        Ok(composite)
    }

    /// Compose a single assignment into a circuit
    async fn compose_assignment(&mut self, assignment: Assignment) -> Result<ComposedCircuit> {
        // Apply qubit mapping
        let mapping = self.mapper.create_mapping(
            &assignment.qubit_mapping,
            &assignment.classical_mapping,
        )?;

        // Apply timing
        let timing = self.timer.create_timing(
            assignment.start_time,
            assignment.duration,
        );

        // For now, create a placeholder circuit
        // In practice, you'd retrieve and transform the original circuit
        let circuit = QuantumCircuit::new(
            format!("job_{}", assignment.job_id),
            assignment.qubit_mapping.len(),
            assignment.classical_mapping.len(),
        );

        Ok(ComposedCircuit {
            job_id: assignment.job_id,
            circuit,
            timing,
            mapping,
        })
    }

    /// Optimize the composite circuit
    async fn optimize_composite(&mut self, composite: &mut CompositeCircuit) -> Result<()> {
        if self.config.optimize_timing {
            self.optimize_timing(composite).await?;
        }

        if self.config.optimize_mapping {
            self.optimize_mappings(composite).await?;
        }

        Ok(())
    }

    /// Optimize timing across circuits
    async fn optimize_timing(&mut self, composite: &mut CompositeCircuit) -> Result<()> {
        // Sort circuits by start time
        composite.circuits.sort_by_key(|c| c.timing.start_time);

        // Try to compress timeline
        let mut current_time = 0;
        for circuit in &mut composite.circuits {
            if circuit.timing.start_time > current_time {
                circuit.timing.start_time = current_time;
            }
            current_time = circuit.timing.start_time + circuit.timing.duration;
        }

        composite.total_duration = current_time;
        Ok(())
    }

    /// Optimize qubit mappings
    async fn optimize_mappings(&mut self, _composite: &mut CompositeCircuit) -> Result<()> {
        // Placeholder for mapping optimization
        Ok(())
    }

    /// Insert buffer operations between circuits
    pub fn insert_buffers(&mut self, composite: &mut CompositeCircuit) -> Result<()> {
        match self.config.buffer_strategy {
            BufferStrategy::None => Ok(()),
            BufferStrategy::Automatic => self.insert_automatic_buffers(composite),
            BufferStrategy::Manual => self.insert_manual_buffers(composite),
            BufferStrategy::Adaptive => self.insert_adaptive_buffers(composite),
        }
    }

    /// Insert reset operations between batches
    pub fn insert_reset_operations(&mut self, composite: &mut CompositeCircuit) -> Result<()> {
        use crate::circuit_ir::Operation;
        
        for i in 1..composite.circuits.len() {
            // Check if circuits share qubits using immutable borrows
            let (share_qubits, shared_qubit_indices) = {
                let current_circuit = &composite.circuits[i];
                let prev_circuit = &composite.circuits[i - 1];
                let share = self.circuits_share_qubits(current_circuit, prev_circuit);
                let indices = if share {
                    self.get_shared_qubits(current_circuit, prev_circuit)
                } else {
                    Vec::new()
                };
                (share, indices)
            };
            
            // Check if circuits share qubits and need reset
            if share_qubits {
                let current_circuit = &mut composite.circuits[i];
                // Add reset operations at the beginning of current circuit
                for qubit_idx in shared_qubit_indices {
                    let reset_op = Operation::Reset {
                        qubit: Qubit(qubit_idx),
                    };
                    current_circuit.circuit.operations.insert(0, reset_op);
                }
            }
        }
        Ok(())
    }

    /// Get qubits shared between two circuits
    fn get_shared_qubits(&self, circuit1: &ComposedCircuit, circuit2: &ComposedCircuit) -> Vec<usize> {
        let qubits1: std::collections::HashSet<_> = circuit1.mapping.qubit_mapping.iter().collect();
        let qubits2: std::collections::HashSet<_> = circuit2.mapping.qubit_mapping.iter().collect();
        qubits1.intersection(&qubits2).map(|&&q| q).collect()
    }

    /// Insert automatic buffers
    fn insert_automatic_buffers(&mut self, composite: &mut CompositeCircuit) -> Result<()> {
        use crate::circuit_ir::Operation;
        use smallvec::SmallVec;
        
        // Add barrier operations between circuits that share qubits
        for i in 0..composite.circuits.len() {
            for j in (i + 1)..composite.circuits.len() {
                if self.circuits_share_qubits(&composite.circuits[i], &composite.circuits[j]) {
                    // Insert barrier operation
                    let shared_qubits = self.get_shared_qubits(&composite.circuits[i], &composite.circuits[j]);
                    if !shared_qubits.is_empty() {
                        let barrier_qubits: SmallVec<[Qubit; 4]> = shared_qubits.into_iter().map(Qubit).collect();
                        let barrier_op = Operation::Barrier {
                            qubits: barrier_qubits,
                        };
                        composite.circuits[j].circuit.operations.insert(0, barrier_op);
                    }
                    
                    // Insert buffer time
                    let buffer_time = 1000; // 1ms buffer
                    if composite.circuits[j].timing.start_time < 
                       composite.circuits[i].timing.start_time + composite.circuits[i].timing.duration + buffer_time {
                        composite.circuits[j].timing.start_time = 
                            composite.circuits[i].timing.start_time + composite.circuits[i].timing.duration + buffer_time;
                    }
                }
            }
        }
        Ok(())
    }

    /// Insert manual buffers
    fn insert_manual_buffers(&mut self, _composite: &mut CompositeCircuit) -> Result<()> {
        // Placeholder for manual buffer insertion
        Ok(())
    }

    /// Insert adaptive buffers
    fn insert_adaptive_buffers(&mut self, composite: &mut CompositeCircuit) -> Result<()> {
        // Use automatic strategy for now
        self.insert_automatic_buffers(composite)
    }

    /// Check if two circuits share qubits
    fn circuits_share_qubits(&self, circuit1: &ComposedCircuit, circuit2: &ComposedCircuit) -> bool {
        let qubits1: std::collections::HashSet<_> = circuit1.mapping.qubit_mapping.iter().collect();
        let qubits2: std::collections::HashSet<_> = circuit2.mapping.qubit_mapping.iter().collect();
        !qubits1.is_disjoint(&qubits2)
    }

    /// Validate the composite circuit
    pub fn validate_composite(&self, composite: &CompositeCircuit) -> Result<ValidationReport> {
        let mut violations = Vec::new();
        let mut warnings = Vec::new();
        
        // Check for resource conflicts
        for i in 0..composite.circuits.len() {
            for j in (i + 1)..composite.circuits.len() {
                let circuit_i = &composite.circuits[i];
                let circuit_j = &composite.circuits[j];
                
                // Check timing conflicts
                if circuit_i.timing.overlaps_with(&circuit_j.timing) {
                    if self.circuits_share_qubits(circuit_i, circuit_j) {
                        violations.push(format!(
                            "Circuits {} and {} have overlapping timing and share resources", 
                            circuit_i.job_id, circuit_j.job_id
                        ));
                    }
                }
                
                // Check for insufficient buffer time
                let gap = circuit_j.timing.gap_to(&circuit_i.timing).abs();
                if gap < 1000 && self.circuits_share_qubits(circuit_i, circuit_j) {
                    warnings.push(format!(
                        "Insufficient buffer time ({} μs) between circuits {} and {}", 
                        gap, circuit_i.job_id, circuit_j.job_id
                    ));
                }
            }
        }
        
        // Check resource utilization
        let utilization = composite.resource_summary.utilization_efficiency;
        if utilization < 0.3 {
            warnings.push(format!(
                "Low resource utilization: {:.1}%", utilization * 100.0
            ));
        }
        
        Ok(ValidationReport {
            is_valid: violations.is_empty(),
            violations,
            warnings,
            total_circuits: composite.circuits.len(),
            total_qubits_used: composite.resource_summary.total_qubits_used,
        })
    }

    /// Get composer statistics
    pub fn statistics(&self) -> ComposerStatistics {
        ComposerStatistics {
            total_compositions: 0, // Would need to track this
            avg_composition_time: 0.0,
            optimization_level: self.config.optimization_level,
            buffer_strategy: self.config.buffer_strategy,
        }
    }
}

/// Composer statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposerStatistics {
    pub total_compositions: usize,
    pub avg_composition_time: f64,
    pub optimization_level: u8,
    pub buffer_strategy: BufferStrategy,
}

/// Validation report for composite circuits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Whether the circuit is valid
    pub is_valid: bool,
    /// Critical validation violations
    pub violations: Vec<String>,
    /// Non-critical warnings
    pub warnings: Vec<String>,
    /// Total number of circuits in composite
    pub total_circuits: usize,
    /// Total qubits used
    pub total_qubits_used: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;
    use crate::scheduler::{Assignment, ScheduleMetadata, ResourceUtilization};

    #[test]
    fn test_composite_circuit_creation() {
        let mut composite = CompositeCircuit::new();
        assert_eq!(composite.circuit_count(), 0);
        assert_eq!(composite.total_duration(), 0);
    }

    #[test]
    fn test_circuit_composer_creation() {
        let topology = TopologyBuilder::grid(3, 3);
        let composer = CircuitComposer::new(topology);
        assert_eq!(composer.config.optimization_level, 2);
    }

    #[tokio::test]
    async fn test_empty_schedule_composition() {
        let topology = TopologyBuilder::grid(2, 2);
        let mut composer = CircuitComposer::new(topology);

        let schedule = Schedule {
            assignments: vec![],
            metadata: ScheduleMetadata::default(),
            total_duration: 0,
            utilization: ResourceUtilization::default(),
        };

        let composite = composer.compose(schedule).await.unwrap();
        assert_eq!(composite.circuit_count(), 0);
    }

    #[tokio::test]
    async fn test_single_assignment_composition() {
        let topology = TopologyBuilder::grid(2, 2);
        let mut composer = CircuitComposer::new(topology);

        let assignment = Assignment::new(0, 1, 1000, 5000)
            .with_qubit_mapping(vec![0, 1])
            .with_classical_mapping(vec![0, 1]);

        let schedule = Schedule {
            assignments: vec![assignment],
            metadata: ScheduleMetadata::default(),
            total_duration: 6000,
            utilization: ResourceUtilization::default(),
        };

        let composite = composer.compose(schedule).await.unwrap();
        assert_eq!(composite.circuit_count(), 1);
        assert_eq!(composite.total_duration(), 6000);
    }
}