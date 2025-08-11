//! Timing management for circuit execution

use crate::{QvmError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Circuit timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitTiming {
    /// Start time (microseconds since epoch)
    pub start_time: u64,
    /// Execution duration (microseconds)
    pub duration: u64,
    /// Estimated end time
    pub estimated_end_time: u64,
}

impl CircuitTiming {
    /// Create new circuit timing
    pub fn new(start_time: u64, duration: u64) -> Self {
        Self {
            start_time,
            duration,
            estimated_end_time: start_time + duration,
        }
    }

    /// Check if this timing overlaps with another
    pub fn overlaps_with(&self, other: &CircuitTiming) -> bool {
        !(self.estimated_end_time <= other.start_time || other.estimated_end_time <= self.start_time)
    }

    /// Get the time gap between this and another timing
    pub fn gap_to(&self, other: &CircuitTiming) -> i64 {
        if self.estimated_end_time <= other.start_time {
            (other.start_time - self.estimated_end_time) as i64
        } else if other.estimated_end_time <= self.start_time {
            (self.start_time - other.estimated_end_time) as i64
        } else {
            0 // Overlapping
        }
    }
}

/// Circuit timer for managing timing operations
#[derive(Debug, Clone)]
pub struct CircuitTimer {
    config: TimerConfig,
}

/// Timer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimerConfig {
    /// Time precision (microseconds)
    pub precision: u64,
    /// Enable timing optimization
    pub optimize_timing: bool,
    /// Buffer time between circuits (microseconds)
    pub buffer_time: u64,
    /// Maximum allowed timing drift
    pub max_drift: u64,
}

impl Default for TimerConfig {
    fn default() -> Self {
        Self {
            precision: 1, // 1 microsecond precision
            optimize_timing: true,
            buffer_time: 1000, // 1ms buffer
            max_drift: 100, // 100μs max drift
        }
    }
}

impl CircuitTimer {
    /// Create a new circuit timer
    pub fn new() -> Self {
        Self {
            config: TimerConfig::default(),
        }
    }

    /// Create timer with custom configuration
    pub fn with_config(config: TimerConfig) -> Self {
        Self { config }
    }

    /// Create timing for a circuit
    pub fn create_timing(&self, start_time: u64, duration: u64) -> CircuitTiming {
        let aligned_start = self.align_time(start_time);
        let aligned_duration = self.align_time(duration);
        
        CircuitTiming::new(aligned_start, aligned_duration)
    }

    /// Align time to precision boundary
    fn align_time(&self, time: u64) -> u64 {
        if self.config.precision <= 1 {
            return time;
        }

        let remainder = time % self.config.precision;
        if remainder == 0 {
            time
        } else {
            time + (self.config.precision - remainder)
        }
    }

    /// Optimize timing for multiple circuits
    pub fn optimize_timings(&self, timings: &mut [CircuitTiming]) -> Result<()> {
        if !self.config.optimize_timing {
            return Ok(());
        }

        // Sort by start time
        timings.sort_by_key(|t| t.start_time);

        // Compress timeline
        self.compress_timeline(timings)?;

        // Add buffer times
        self.add_buffer_times(timings)?;

        Ok(())
    }

    /// Compress timeline to minimize total execution time
    fn compress_timeline(&self, timings: &mut [CircuitTiming]) -> Result<()> {
        let mut current_time = 0;

        for timing in timings.iter_mut() {
            if timing.start_time > current_time {
                timing.start_time = current_time;
                timing.estimated_end_time = current_time + timing.duration;
            }
            current_time = timing.estimated_end_time;
        }

        Ok(())
    }

    /// Add buffer times between circuits
    fn add_buffer_times(&self, timings: &mut [CircuitTiming]) -> Result<()> {
        for i in 1..timings.len() {
            let prev_end = timings[i - 1].estimated_end_time;
            let current_start = timings[i].start_time;
            
            if current_start < prev_end + self.config.buffer_time {
                let new_start = prev_end + self.config.buffer_time;
                timings[i].start_time = new_start;
                timings[i].estimated_end_time = new_start + timings[i].duration;
            }
        }

        Ok(())
    }

    /// Validate timing constraints
    pub fn validate_timings(&self, timings: &[CircuitTiming]) -> Result<Vec<TimingViolation>> {
        let mut violations = Vec::new();

        // Check for overlaps
        for i in 0..timings.len() {
            for j in (i + 1)..timings.len() {
                if timings[i].overlaps_with(&timings[j]) {
                    violations.push(TimingViolation {
                        violation_type: ViolationType::Overlap,
                        circuit1: i,
                        circuit2: Some(j),
                        description: format!("Circuits {} and {} have overlapping execution times", i, j),
                        severity: Severity::Critical,
                    });
                }
            }
        }

        // Check buffer time violations
        for i in 1..timings.len() {
            let gap = timings[i].start_time.saturating_sub(timings[i - 1].estimated_end_time);
            if gap < self.config.buffer_time {
                violations.push(TimingViolation {
                    violation_type: ViolationType::InsufficientBuffer,
                    circuit1: i - 1,
                    circuit2: Some(i),
                    description: format!("Insufficient buffer time between circuits {} and {}", i - 1, i),
                    severity: Severity::Warning,
                });
            }
        }

        Ok(violations)
    }

    /// Calculate timing statistics
    pub fn calculate_statistics(&self, timings: &[CircuitTiming]) -> TimingStatistics {
        if timings.is_empty() {
            return TimingStatistics::default();
        }

        let total_duration = timings.iter().map(|t| t.duration).sum();
        let total_execution_time = timings.iter()
            .map(|t| t.estimated_end_time)
            .max()
            .unwrap_or(0);

        let gaps: Vec<u64> = (1..timings.len())
            .map(|i| {
                timings[i].start_time.saturating_sub(timings[i - 1].estimated_end_time)
            })
            .collect();

        let total_gap_time: u64 = gaps.iter().sum();
        let avg_gap = if gaps.is_empty() { 0.0 } else { total_gap_time as f64 / gaps.len() as f64 };

        let efficiency = if total_execution_time > 0 {
            total_duration as f64 / total_execution_time as f64
        } else {
            0.0
        };

        TimingStatistics {
            total_circuits: timings.len(),
            total_duration,
            total_execution_time,
            total_gap_time,
            average_gap: avg_gap,
            efficiency,
            compression_ratio: if total_duration > 0 {
                total_execution_time as f64 / total_duration as f64
            } else {
                1.0
            },
        }
    }
}

impl Default for CircuitTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Timing violation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingViolation {
    /// Type of violation
    pub violation_type: ViolationType,
    /// First circuit involved
    pub circuit1: usize,
    /// Second circuit involved (if applicable)
    pub circuit2: Option<usize>,
    /// Description of the violation
    pub description: String,
    /// Severity level
    pub severity: Severity,
}

/// Types of timing violations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationType {
    /// Circuits have overlapping execution times
    Overlap,
    /// Insufficient buffer time between circuits
    InsufficientBuffer,
    /// Timing precision violation
    PrecisionViolation,
    /// Excessive timing drift
    ExcessiveDrift,
}

/// Violation severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    /// Critical - execution will fail
    Critical,
    /// Warning - execution may have issues
    Warning,
    /// Info - minor optimization opportunity
    Info,
}

/// Timing statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TimingStatistics {
    /// Total number of circuits
    pub total_circuits: usize,
    /// Total duration of all circuits
    pub total_duration: u64,
    /// Total execution time (including gaps)
    pub total_execution_time: u64,
    /// Total gap time between circuits
    pub total_gap_time: u64,
    /// Average gap between circuits
    pub average_gap: f64,
    /// Timing efficiency (active time / total time)
    pub efficiency: f64,
    /// Compression ratio (how much timeline was compressed)
    pub compression_ratio: f64,
}

/// Advanced timing optimizer
pub struct TimingOptimizer;

impl TimingOptimizer {
    /// Optimize timing using advanced algorithms
    pub fn optimize_advanced(
        timings: &mut [CircuitTiming],
        constraints: &TimingConstraints,
    ) -> Result<OptimizerResult> {
        let original_stats = Self::calculate_stats(timings);

        // Apply various optimization techniques
        Self::apply_parallelization(timings, constraints)?;
        Self::apply_reordering(timings, constraints)?;
        Self::apply_compression(timings, constraints)?;

        let optimized_stats = Self::calculate_stats(timings);

        Ok(OptimizerResult {
            original_duration: original_stats.total_execution_time,
            optimized_duration: optimized_stats.total_execution_time,
            improvement: (original_stats.total_execution_time as f64 - optimized_stats.total_execution_time as f64) 
                        / original_stats.total_execution_time as f64,
            techniques_applied: vec!["parallelization".to_string(), "reordering".to_string(), "compression".to_string()],
        })
    }

    /// Apply parallelization optimization
    fn apply_parallelization(
        timings: &mut [CircuitTiming],
        _constraints: &TimingConstraints,
    ) -> Result<()> {
        // Group non-conflicting circuits for parallel execution
        // This is a simplified implementation
        for i in 1..timings.len() {
            // If circuits don't share resources, they can run in parallel
            if timings[i].start_time > timings[i - 1].start_time + timings[i - 1].duration / 2 {
                timings[i].start_time = timings[i - 1].start_time;
                timings[i].estimated_end_time = timings[i].start_time + timings[i].duration;
            }
        }
        Ok(())
    }

    /// Apply reordering optimization
    fn apply_reordering(
        timings: &mut [CircuitTiming],
        _constraints: &TimingConstraints,
    ) -> Result<()> {
        // Sort by duration (shortest first) for better packing
        timings.sort_by_key(|t| t.duration);
        
        // Reassign start times
        let mut current_time = 0;
        for timing in timings.iter_mut() {
            timing.start_time = current_time;
            timing.estimated_end_time = current_time + timing.duration;
            current_time += timing.duration;
        }
        
        Ok(())
    }

    /// Apply compression optimization
    fn apply_compression(
        timings: &mut [CircuitTiming],
        constraints: &TimingConstraints,
    ) -> Result<()> {
        let min_gap = constraints.min_gap_time;
        
        // Remove excessive gaps
        for i in 1..timings.len() {
            let actual_gap = timings[i].start_time.saturating_sub(timings[i - 1].estimated_end_time);
            if actual_gap > min_gap {
                let excess = actual_gap - min_gap;
                timings[i].start_time -= excess;
                timings[i].estimated_end_time -= excess;
            }
        }
        
        Ok(())
    }

    /// Calculate basic statistics
    fn calculate_stats(timings: &[CircuitTiming]) -> TimingStatistics {
        let timer = CircuitTimer::new();
        timer.calculate_statistics(timings)
    }
}

/// Timing constraints for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConstraints {
    /// Minimum gap time between circuits
    pub min_gap_time: u64,
    /// Maximum allowed total duration
    pub max_total_duration: Option<u64>,
    /// Resource conflict information
    pub resource_conflicts: HashMap<(usize, usize), bool>,
    /// Priority weights for circuits
    pub circuit_priorities: Vec<f64>,
}

impl Default for TimingConstraints {
    fn default() -> Self {
        Self {
            min_gap_time: 1000, // 1ms
            max_total_duration: None,
            resource_conflicts: HashMap::new(),
            circuit_priorities: Vec::new(),
        }
    }
}

/// Result of timing optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerResult {
    /// Original total duration
    pub original_duration: u64,
    /// Optimized total duration
    pub optimized_duration: u64,
    /// Improvement ratio (0.0 to 1.0)
    pub improvement: f64,
    /// List of optimization techniques applied
    pub techniques_applied: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_timing_creation() {
        let timing = CircuitTiming::new(1000, 5000);
        assert_eq!(timing.start_time, 1000);
        assert_eq!(timing.duration, 5000);
        assert_eq!(timing.estimated_end_time, 6000);
    }

    #[test]
    fn test_timing_overlap() {
        let timing1 = CircuitTiming::new(1000, 3000);
        let timing2 = CircuitTiming::new(2000, 3000);
        let timing3 = CircuitTiming::new(5000, 1000);

        assert!(timing1.overlaps_with(&timing2));
        assert!(!timing1.overlaps_with(&timing3));
    }

    #[test]
    fn test_circuit_timer() {
        let timer = CircuitTimer::new();
        let timing = timer.create_timing(1001, 2001);
        
        // Should align to precision boundary
        assert_eq!(timing.start_time, 1001);
        assert_eq!(timing.duration, 2001);
    }

    #[test]
    fn test_timing_optimization() {
        let timer = CircuitTimer::new();
        let mut timings = vec![
            CircuitTiming::new(1000, 2000),
            CircuitTiming::new(5000, 1000), // Large gap
            CircuitTiming::new(8000, 1500),
        ];

        timer.optimize_timings(&mut timings).unwrap();
        
        // Should compress timeline
        assert_eq!(timings[0].start_time, 0);
        assert!(timings[1].start_time >= timings[0].estimated_end_time);
    }

    #[test]
    fn test_timing_validation() {
        let timer = CircuitTimer::new();
        let timings = vec![
            CircuitTiming::new(1000, 2000),
            CircuitTiming::new(1500, 1000), // Overlaps with first
        ];

        let violations = timer.validate_timings(&timings).unwrap();
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].violation_type, ViolationType::Overlap);
    }

    #[test]
    fn test_timing_statistics() {
        let timer = CircuitTimer::new();
        let timings = vec![
            CircuitTiming::new(0, 1000),
            CircuitTiming::new(2000, 1500), // 1000μs gap
            CircuitTiming::new(4000, 500),  // 500μs gap
        ];

        let stats = timer.calculate_statistics(&timings);
        assert_eq!(stats.total_circuits, 3);
        assert_eq!(stats.total_duration, 3000);
        assert_eq!(stats.total_execution_time, 4500);
        assert_eq!(stats.total_gap_time, 1500);
    }
}