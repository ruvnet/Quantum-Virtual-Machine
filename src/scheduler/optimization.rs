//! Scheduling optimization algorithms for minimal batch count and improved efficiency

use crate::{QvmError, Result, Topology};
use crate::scheduler::{Job, Assignment, Schedule, BinPacker, BinPackingAlgorithm};
use crate::scheduler::batch::{Batch, BatchConfig, BatchScheduler};
use crate::scheduler::multiplexing::{SpatialMultiplexer, TemporalMultiplexer, HybridMultiplexer};
use crate::scheduler::multiplexing::{SpatialMultiplexConfig, TemporalMultiplexConfig};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokio::task;
use tokio::time::{Duration, Instant};

/// Optimization configuration for scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Target to minimize
    pub objective: OptimizationObjective,
    /// Maximum optimization time (milliseconds)
    pub max_optimization_time: u64,
    /// Optimization algorithm to use
    pub algorithm: OptimizationAlgorithm,
    /// Enable parallel optimization
    pub enable_parallel: bool,
    /// Number of optimization iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            objective: OptimizationObjective::MinimizeBatchCount,
            max_optimization_time: 5000, // 5 seconds
            algorithm: OptimizationAlgorithm::SimulatedAnnealing,
            enable_parallel: true,
            max_iterations: 1000,
            convergence_threshold: 0.001,
        }
    }
}

/// Optimization objectives
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimize the total number of batches
    MinimizeBatchCount,
    /// Minimize the total execution time
    MinimizeExecutionTime,
    /// Maximize resource utilization
    MaximizeUtilization,
    /// Minimize energy consumption
    MinimizeEnergy,
    /// Balance multiple objectives
    MultiObjective,
}

/// Optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    /// Simulated annealing
    SimulatedAnnealing,
    /// Genetic algorithm
    GeneticAlgorithm,
    /// Greedy local search
    GreedySearch,
    /// Tabu search
    TabuSearch,
    /// Particle swarm optimization
    ParticleSwarm,
}

/// Scheduling optimizer
#[derive(Debug, Clone)]
pub struct SchedulingOptimizer {
    topology: Topology,
    config: OptimizationConfig,
}

impl SchedulingOptimizer {
    /// Create a new scheduling optimizer
    pub fn new(topology: Topology, config: OptimizationConfig) -> Self {
        Self { topology, config }
    }

    /// Optimize a schedule to minimize the objective function
    pub async fn optimize_schedule(&self, jobs: Vec<Job>) -> Result<Schedule> {
        let start_time = Instant::now();
        let timeout = Duration::from_millis(self.config.max_optimization_time);

        match self.config.algorithm {
            OptimizationAlgorithm::SimulatedAnnealing => {
                self.simulated_annealing_optimization(jobs, start_time, timeout).await
            }
            OptimizationAlgorithm::GeneticAlgorithm => {
                self.genetic_algorithm_optimization(jobs, start_time, timeout).await
            }
            OptimizationAlgorithm::GreedySearch => {
                self.greedy_search_optimization(jobs, start_time, timeout).await
            }
            OptimizationAlgorithm::TabuSearch => {
                self.tabu_search_optimization(jobs, start_time, timeout).await
            }
            OptimizationAlgorithm::ParticleSwarm => {
                self.particle_swarm_optimization(jobs, start_time, timeout).await
            }
        }
    }

    /// Simulated annealing optimization
    async fn simulated_annealing_optimization(
        &self,
        jobs: Vec<Job>,
        start_time: Instant,
        timeout: Duration,
    ) -> Result<Schedule> {
        // Initial solution using greedy approach
        let mut current_solution = self.create_initial_solution(jobs.clone()).await?;
        let mut current_cost = self.evaluate_solution(&current_solution);
        let mut best_solution = current_solution.clone();
        let mut best_cost = current_cost;

        // Simulated annealing parameters
        let mut temperature = 1000.0;
        let cooling_rate = 0.995;
        let min_temperature = 1.0;

        for iteration in 0..self.config.max_iterations {
            if start_time.elapsed() > timeout {
                break;
            }

            // Generate neighbor solution
            let neighbor_solution = self.generate_neighbor_solution(&current_solution).await?;
            let neighbor_cost = self.evaluate_solution(&neighbor_solution);

            // Accept or reject the neighbor solution
            let cost_diff = neighbor_cost - current_cost;
            let accept_probability = if cost_diff < 0.0 {
                1.0
            } else {
                (-cost_diff / temperature).exp()
            };

            if accept_probability > rand::random::<f64>() {
                current_solution = neighbor_solution;
                current_cost = neighbor_cost;

                // Update best solution
                if current_cost < best_cost {
                    best_solution = current_solution.clone();
                    best_cost = current_cost;
                }
            }

            // Cool down temperature
            temperature *= cooling_rate;
            if temperature < min_temperature {
                temperature = min_temperature;
            }

            // Check convergence
            if iteration > 100 && (best_cost - current_cost).abs() < self.config.convergence_threshold {
                break;
            }
        }

        Ok(best_solution)
    }

    /// Genetic algorithm optimization
    async fn genetic_algorithm_optimization(
        &self,
        jobs: Vec<Job>,
        start_time: Instant,
        timeout: Duration,
    ) -> Result<Schedule> {
        let population_size = 50;
        let mutation_rate = 0.1;
        let crossover_rate = 0.8;
        let elite_size = 10;

        // Initialize population
        let mut population = Vec::new();
        for _ in 0..population_size {
            let individual = self.create_random_solution(&jobs).await?;
            population.push(individual);
        }

        for generation in 0..self.config.max_iterations {
            if start_time.elapsed() > timeout {
                break;
            }

            // Evaluate fitness
            let fitness_scores: Vec<f64> = population
                .iter()
                .map(|individual| 1.0 / (1.0 + self.evaluate_solution(individual)))
                .collect();

            // Selection
            let mut new_population = Vec::new();
            
            // Keep elite individuals
            let mut sorted_indices: Vec<usize> = (0..population_size).collect();
            sorted_indices.sort_by(|&a, &b| fitness_scores[b].partial_cmp(&fitness_scores[a]).unwrap());
            
            for i in 0..elite_size {
                new_population.push(population[sorted_indices[i]].clone());
            }

            // Generate offspring
            while new_population.len() < population_size {
                if rand::random::<f64>() < crossover_rate {
                    let parent1_idx = self.tournament_selection(&fitness_scores);
                    let parent2_idx = self.tournament_selection(&fitness_scores);
                    
                    if let Ok(offspring) = self.crossover(&population[parent1_idx], &population[parent2_idx]).await {
                        new_population.push(offspring);
                    } else {
                        new_population.push(population[parent1_idx].clone());
                    }
                } else {
                    let parent_idx = self.tournament_selection(&fitness_scores);
                    new_population.push(population[parent_idx].clone());
                }

                // Mutation
                if let Some(last) = new_population.last_mut() {
                    if rand::random::<f64>() < mutation_rate {
                        if let Ok(mutated) = self.mutate(last).await {
                            *last = mutated;
                        }
                    }
                }
            }

            population = new_population;
        }

        // Return best individual
        let fitness_scores: Vec<f64> = population
            .iter()
            .map(|individual| 1.0 / (1.0 + self.evaluate_solution(individual)))
            .collect();
        
        let best_idx = fitness_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok(population[best_idx].clone())
    }

    /// Greedy search optimization
    async fn greedy_search_optimization(
        &self,
        jobs: Vec<Job>,
        start_time: Instant,
        timeout: Duration,
    ) -> Result<Schedule> {
        let mut best_solution = self.create_initial_solution(jobs.clone()).await?;
        let mut best_cost = self.evaluate_solution(&best_solution);

        for _ in 0..self.config.max_iterations {
            if start_time.elapsed() > timeout {
                break;
            }

            // Try all possible local improvements
            let neighbors = self.generate_all_neighbors(&best_solution).await?;
            
            let mut improved = false;
            for neighbor in neighbors {
                let neighbor_cost = self.evaluate_solution(&neighbor);
                if neighbor_cost < best_cost {
                    best_solution = neighbor;
                    best_cost = neighbor_cost;
                    improved = true;
                    break;
                }
            }

            if !improved {
                break; // Local optimum reached
            }
        }

        Ok(best_solution)
    }

    /// Tabu search optimization
    async fn tabu_search_optimization(
        &self,
        jobs: Vec<Job>,
        start_time: Instant,
        timeout: Duration,
    ) -> Result<Schedule> {
        let mut current_solution = self.create_initial_solution(jobs.clone()).await?;
        let mut best_solution = current_solution.clone();
        let mut best_cost = self.evaluate_solution(&best_solution);
        
        let mut tabu_list: Vec<u64> = Vec::new();
        let tabu_tenure = 7;
        let max_tabu_size = 100;

        for _ in 0..self.config.max_iterations {
            if start_time.elapsed() > timeout {
                break;
            }

            let neighbors = self.generate_all_neighbors(&current_solution).await?;
            let mut best_neighbor = None;
            let mut best_neighbor_cost = f64::INFINITY;

            for neighbor in neighbors {
                let neighbor_hash = self.hash_solution(&neighbor);
                let neighbor_cost = self.evaluate_solution(&neighbor);

                // Skip if in tabu list unless it's better than best known
                if tabu_list.contains(&neighbor_hash) && neighbor_cost >= best_cost {
                    continue;
                }

                if neighbor_cost < best_neighbor_cost {
                    best_neighbor = Some(neighbor);
                    best_neighbor_cost = neighbor_cost;
                }
            }

            if let Some(neighbor) = best_neighbor {
                current_solution = neighbor;
                let current_hash = self.hash_solution(&current_solution);
                
                // Add to tabu list
                tabu_list.push(current_hash);
                if tabu_list.len() > max_tabu_size {
                    tabu_list.remove(0);
                }

                // Update best solution
                if best_neighbor_cost < best_cost {
                    best_solution = current_solution.clone();
                    best_cost = best_neighbor_cost;
                }
            } else {
                break; // No valid neighbors
            }
        }

        Ok(best_solution)
    }

    /// Particle swarm optimization
    async fn particle_swarm_optimization(
        &self,
        jobs: Vec<Job>,
        start_time: Instant,
        timeout: Duration,
    ) -> Result<Schedule> {
        // For scheduling problems, PSO needs to be adapted
        // This is a simplified version that uses multiple random restarts
        let num_particles = 20;
        let mut best_global_solution = self.create_initial_solution(jobs.clone()).await?;
        let mut best_global_cost = self.evaluate_solution(&best_global_solution);

        for _ in 0..num_particles {
            if start_time.elapsed() > timeout {
                break;
            }

            let particle_solution = self.create_random_solution(&jobs).await?;
            let particle_cost = self.evaluate_solution(&particle_solution);

            if particle_cost < best_global_cost {
                best_global_solution = particle_solution;
                best_global_cost = particle_cost;
            }
        }

        // Local improvement on best solution
        let improved_solution = self.local_improvement(best_global_solution).await?;
        Ok(improved_solution)
    }

    /// Create initial solution using best available method
    async fn create_initial_solution(&self, jobs: Vec<Job>) -> Result<Schedule> {
        // Use hybrid multiplexing for initial solution
        let spatial_config = SpatialMultiplexConfig::default();
        let temporal_config = TemporalMultiplexConfig::default();
        let mut hybrid_multiplexer = HybridMultiplexer::new(
            self.topology.clone(),
            spatial_config,
            temporal_config,
        );

        let assignments = hybrid_multiplexer.schedule_hybrid(jobs).await?;
        
        Ok(Schedule {
            assignments,
            metadata: Default::default(),
            total_duration: 0,
            utilization: Default::default(),
        })
    }

    /// Create a random solution
    async fn create_random_solution(&self, jobs: &[Job]) -> Result<Schedule> {
        let mut shuffled_jobs = jobs.to_vec();
        use rand::seq::SliceRandom;
        {
            let mut rng = rand::thread_rng();
            shuffled_jobs.shuffle(&mut rng);
        } // rng goes out of scope here, before await

        self.create_initial_solution(shuffled_jobs).await
    }

    /// Evaluate solution quality based on objective
    fn evaluate_solution(&self, schedule: &Schedule) -> f64 {
        match self.config.objective {
            OptimizationObjective::MinimizeBatchCount => {
                self.count_batches(schedule) as f64
            }
            OptimizationObjective::MinimizeExecutionTime => {
                schedule.total_duration as f64
            }
            OptimizationObjective::MaximizeUtilization => {
                1.0 - schedule.utilization.avg_qubit_utilization
            }
            OptimizationObjective::MinimizeEnergy => {
                self.estimate_energy_consumption(schedule)
            }
            OptimizationObjective::MultiObjective => {
                // Weighted combination of objectives
                let batch_count = self.count_batches(schedule) as f64;
                let execution_time = schedule.total_duration as f64;
                let utilization = 1.0 - schedule.utilization.avg_qubit_utilization;
                
                0.4 * batch_count + 0.3 * (execution_time / 1000000.0) + 0.3 * utilization
            }
        }
    }

    /// Count number of temporal batches in schedule
    fn count_batches(&self, schedule: &Schedule) -> usize {
        let mut time_slots = HashSet::new();
        for assignment in &schedule.assignments {
            time_slots.insert(assignment.start_time);
        }
        time_slots.len()
    }

    /// Estimate energy consumption
    fn estimate_energy_consumption(&self, schedule: &Schedule) -> f64 {
        // Simplified energy model: energy = power * time
        let base_power = 1.0; // Watts per qubit
        let mut total_energy = 0.0;

        for assignment in &schedule.assignments {
            let power = base_power * assignment.qubit_mapping.len() as f64;
            let time_seconds = assignment.duration as f64 / 1_000_000.0;
            total_energy += power * time_seconds;
        }

        total_energy
    }

    /// Generate a neighbor solution
    async fn generate_neighbor_solution(&self, solution: &Schedule) -> Result<Schedule> {
        // Simple neighborhood operation: swap two assignments
        let mut new_solution = solution.clone();
        
        if new_solution.assignments.len() >= 2 {
            let idx1 = rand::random::<usize>() % new_solution.assignments.len();
            let idx2 = rand::random::<usize>() % new_solution.assignments.len();
            
            if idx1 != idx2 {
                new_solution.assignments.swap(idx1, idx2);
                // Update timings
                self.update_timings(&mut new_solution);
            }
        }

        Ok(new_solution)
    }

    /// Generate all neighbor solutions
    async fn generate_all_neighbors(&self, solution: &Schedule) -> Result<Vec<Schedule>> {
        let mut neighbors = Vec::new();
        
        // Try all possible swaps
        for i in 0..solution.assignments.len() {
            for j in (i + 1)..solution.assignments.len() {
                let mut neighbor = solution.clone();
                neighbor.assignments.swap(i, j);
                self.update_timings(&mut neighbor);
                neighbors.push(neighbor);
            }
        }

        Ok(neighbors)
    }

    /// Tournament selection for genetic algorithm
    fn tournament_selection(&self, fitness_scores: &[f64]) -> usize {
        let tournament_size = 3;
        let mut best_idx = rand::random::<usize>() % fitness_scores.len();
        let mut best_fitness = fitness_scores[best_idx];

        for _ in 1..tournament_size {
            let candidate_idx = rand::random::<usize>() % fitness_scores.len();
            if fitness_scores[candidate_idx] > best_fitness {
                best_idx = candidate_idx;
                best_fitness = fitness_scores[candidate_idx];
            }
        }

        best_idx
    }

    /// Crossover operation for genetic algorithm
    async fn crossover(&self, parent1: &Schedule, parent2: &Schedule) -> Result<Schedule> {
        // Order crossover: take a segment from parent1 and fill rest from parent2
        if parent1.assignments.is_empty() || parent2.assignments.is_empty() {
            return Ok(parent1.clone());
        }

        let len = parent1.assignments.len().min(parent2.assignments.len());
        let crossover_point = rand::random::<usize>() % len;
        
        let mut child_assignments = Vec::new();
        
        // Take first part from parent1
        child_assignments.extend(parent1.assignments[..crossover_point].iter().cloned());
        
        // Fill rest from parent2 (avoiding duplicates by job_id)
        let used_jobs: HashSet<usize> = child_assignments.iter().map(|a| a.job_id).collect();
        for assignment in &parent2.assignments {
            if !used_jobs.contains(&assignment.job_id) {
                child_assignments.push(assignment.clone());
            }
        }

        let mut child = Schedule {
            assignments: child_assignments,
            metadata: Default::default(),
            total_duration: 0,
            utilization: Default::default(),
        };

        self.update_timings(&mut child);
        Ok(child)
    }

    /// Mutation operation for genetic algorithm
    async fn mutate(&self, solution: &Schedule) -> Result<Schedule> {
        self.generate_neighbor_solution(solution).await
    }

    /// Local improvement using hill climbing
    async fn local_improvement(&self, mut solution: Schedule) -> Result<Schedule> {
        let mut improved = true;
        let mut current_cost = self.evaluate_solution(&solution);

        while improved {
            improved = false;
            
            if let Ok(neighbors) = self.generate_all_neighbors(&solution).await {
                for neighbor in neighbors {
                    let neighbor_cost = self.evaluate_solution(&neighbor);
                    if neighbor_cost < current_cost {
                        solution = neighbor;
                        current_cost = neighbor_cost;
                        improved = true;
                        break;
                    }
                }
            }
        }

        Ok(solution)
    }

    /// Hash a solution for tabu search
    fn hash_solution(&self, solution: &Schedule) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for assignment in &solution.assignments {
            assignment.job_id.hash(&mut hasher);
            assignment.start_time.hash(&mut hasher);
            assignment.tile_id.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Update timing information in a schedule
    fn update_timings(&self, schedule: &mut Schedule) {
        // Recalculate total duration and utilization
        schedule.total_duration = schedule.assignments
            .iter()
            .map(|a| a.start_time + a.duration)
            .max()
            .unwrap_or(0);

        // Update utilization metrics (simplified)
        let total_qubits = self.topology.qubit_count();
        let total_qubit_time: u64 = schedule.assignments
            .iter()
            .map(|a| a.qubit_mapping.len() as u64 * a.duration)
            .sum();

        let total_possible_time = total_qubits as u64 * schedule.total_duration;
        schedule.utilization.avg_qubit_utilization = if total_possible_time > 0 {
            total_qubit_time as f64 / total_possible_time as f64
        } else {
            0.0
        };
    }
}

/// Parallel optimization coordinator
pub struct ParallelOptimizer {
    topology: Topology,
    config: OptimizationConfig,
}

impl ParallelOptimizer {
    pub fn new(topology: Topology, config: OptimizationConfig) -> Self {
        Self { topology, config }
    }

    /// Run multiple optimization algorithms in parallel and return the best result
    pub async fn optimize_parallel(&self, jobs: Vec<Job>) -> Result<Schedule> {
        if !self.config.enable_parallel {
            let optimizer = SchedulingOptimizer::new(self.topology.clone(), self.config.clone());
            return optimizer.optimize_schedule(jobs).await;
        }

        // Create multiple optimizers with different algorithms
        let algorithms = vec![
            OptimizationAlgorithm::SimulatedAnnealing,
            OptimizationAlgorithm::GeneticAlgorithm,
            OptimizationAlgorithm::GreedySearch,
            OptimizationAlgorithm::TabuSearch,
        ];

        let mut handles = Vec::new();
        
        for algorithm in algorithms {
            let jobs_clone = jobs.clone();
            let topology_clone = self.topology.clone();
            let mut config_clone = self.config.clone();
            config_clone.algorithm = algorithm;
            
            let handle = task::spawn(async move {
                let optimizer = SchedulingOptimizer::new(topology_clone, config_clone);
                optimizer.optimize_schedule(jobs_clone).await
            });
            
            handles.push(handle);
        }

        // Wait for all optimizers to complete and find the best result
        let mut best_solution = None;
        let mut best_cost = f64::INFINITY;
        let temp_optimizer = SchedulingOptimizer::new(self.topology.clone(), self.config.clone());

        for handle in handles {
            if let Ok(Ok(solution)) = handle.await {
                let cost = temp_optimizer.evaluate_solution(&solution);
                if cost < best_cost {
                    best_cost = cost;
                    best_solution = Some(solution);
                }
            }
        }

        best_solution.ok_or_else(|| QvmError::scheduling_error("All optimization attempts failed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::topology::TopologyBuilder;
    use crate::circuit_ir::CircuitBuilder;

    #[test]
    fn test_optimization_config() {
        let config = OptimizationConfig::default();
        assert_eq!(config.objective, OptimizationObjective::MinimizeBatchCount);
        assert_eq!(config.algorithm, OptimizationAlgorithm::SimulatedAnnealing);
        assert!(config.enable_parallel);
    }

    #[tokio::test]
    async fn test_scheduling_optimizer() {
        let topology = TopologyBuilder::grid(4, 4);
        let config = OptimizationConfig::default();
        let optimizer = SchedulingOptimizer::new(topology, config);

        let circuit = CircuitBuilder::new("test", 2, 2).h(0).unwrap().build();
        let jobs = vec![
            Job::new(0, circuit.clone()),
            Job::new(1, circuit.clone()),
            Job::new(2, circuit),
        ];

        let schedule = optimizer.optimize_schedule(jobs).await.unwrap();
        assert_eq!(schedule.assignments.len(), 3);
    }

    #[tokio::test]
    async fn test_parallel_optimizer() {
        let topology = TopologyBuilder::grid(3, 3);
        let mut config = OptimizationConfig::default();
        config.max_optimization_time = 1000; // 1 second
        config.max_iterations = 10; // Reduced for testing
        
        let optimizer = ParallelOptimizer::new(topology, config);

        let circuit = CircuitBuilder::new("test", 2, 2).h(0).unwrap().build();
        let jobs = vec![Job::new(0, circuit), Job::new(1, CircuitBuilder::new("test2", 3, 3).h(0).unwrap().build())];

        let schedule = optimizer.optimize_parallel(jobs).await.unwrap();
        assert_eq!(schedule.assignments.len(), 2);
    }
}