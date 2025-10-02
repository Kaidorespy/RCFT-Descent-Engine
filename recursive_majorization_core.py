"""
RCFT Section 1: Recursive Majorization Core
==========================================

Implementing the fundamental RCFT extension to classical majorization:
- EchoMemoryManager: Tracks partition transition history with exponential decay
- RecursiveMajorizationAnalyzer: Extends classical analysis with memory-aware ordering
- Recursive comparator kernel: Implements λ ≻ᵣ μ operator

"Memory is not added to chaos—it is extracted from it."
"""

import numpy as np
from collections import defaultdict, deque
from math import exp, sqrt
from typing import Dict, List, Tuple, Optional, Any
import time
import json
from dataclasses import dataclass, asdict
from copy import deepcopy

@dataclass
class TransitionRecord:
    """Single transition memory record"""
    timestamp: float
    delta_S: float
    delta_I: float
    weight: float
    source_partition: Tuple[int, ...]
    target_partition: Tuple[int, ...]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TransitionRecord':
        """Create from dictionary"""
        return cls(**data)


class EchoMemoryManager:
    """
    Manages memory fields for partitions with exponential decay.
    
    Each partition λ accumulates a memory vector E(λ) = [mean ΔS, std ΔS, mean ΔI, std ΔI]
    computed from exponentially weighted recent transitions TO that partition.
    """
    
    def __init__(self, tau: float = 5.0, min_history: int = 3, max_history: int = 50):
        """
        Initialize memory manager
        
        Args:
            tau: Decay time constant for exponential weighting
            min_history: Minimum transitions before computing echo vectors
            max_history: Maximum history length to store (memory efficiency)
        """
        self.tau = tau
        self.min_history = min_history
        self.max_history = max_history
        
        # Memory storage: partition_tuple -> List[TransitionRecord]
        self.memory: Dict[Tuple[int, ...], List[TransitionRecord]] = defaultdict(list)
        
        # Cached echo vectors: partition_tuple -> np.array([mean_dS, std_dS, mean_dI, std_dI])
        self._echo_cache: Dict[Tuple[int, ...], np.ndarray] = {}
        self._cache_valid: Dict[Tuple[int, ...], bool] = defaultdict(bool)
        
        # Global entropy tracking for adaptive thresholding
        self.global_entropy_history: deque = deque(maxlen=25)
        self.current_time = 0.0
    
    def record_transition(self, 
                         source_partition: Tuple[int, ...],
                         target_partition: Tuple[int, ...],
                         delta_S: float,
                         delta_I: float) -> None:
        """
        Record a partition-to-partition transition
        
        Args:
            source_partition: Origin partition
            target_partition: Destination partition  
            delta_S: Change in entropy (target - source)
            delta_I: Change in incomparability (target - source)
        """
        self.current_time += 1.0
        
        # Create transition record
        record = TransitionRecord(
            timestamp=self.current_time,
            delta_S=delta_S,
            delta_I=delta_I,
            weight=1.0,  # Will be recomputed when calculating echo vector
            source_partition=source_partition,
            target_partition=target_partition
        )
        
        # Store in target partition's memory (transitions TO this partition)
        self.memory[target_partition].append(record)
        
        # Trim memory if too long
        if len(self.memory[target_partition]) > self.max_history:
            self.memory[target_partition] = self.memory[target_partition][-self.max_history:]
        
        # Invalidate cache for target partition
        self._cache_valid[target_partition] = False
        
        # Update global entropy tracking
        if hasattr(self, '_current_global_entropy'):
            self.global_entropy_history.append(self._current_global_entropy)
    
    def update_global_entropy(self, entropy: float) -> None:
        """Update global entropy for adaptive thresholding"""
        self._current_global_entropy = entropy
    
    def _compute_decay_weights(self, records: List[TransitionRecord]) -> np.ndarray:
        """Compute exponential decay weights for transition records"""
        if not records:
            return np.array([])
        
        current_time = self.current_time
        timestamps = np.array([r.timestamp for r in records])
        delta_times = current_time - timestamps
        
        # Exponential decay: w = e^(-Δt/τ)
        weights = np.exp(-delta_times / self.tau)
        
        # Normalize weights to sum to 1
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        
        return weights
    
    def get_echo_vector(self, partition: Tuple[int, ...]) -> Optional[np.ndarray]:
        """
        Compute 4D echo vector [mean ΔS, std ΔS, mean ΔI, std ΔI] for partition
        
        Args:
            partition: Partition tuple
            
        Returns:
            4D numpy array or None if insufficient history
        """
        if partition not in self.memory:
            return None
        
        records = self.memory[partition]
        
        # Check minimum history requirement
        if len(records) < self.min_history:
            return None
        
        # Use cached vector if valid
        if self._cache_valid.get(partition, False):
            return self._echo_cache.get(partition, None)
        
        # Compute decay weights
        weights = self._compute_decay_weights(records)
        
        if len(weights) == 0:
            return None
        
        # Extract deltas
        delta_S_values = np.array([r.delta_S for r in records])
        delta_I_values = np.array([r.delta_I for r in records])
        
        # Compute weighted statistics
        mean_dS = np.average(delta_S_values, weights=weights)
        mean_dI = np.average(delta_I_values, weights=weights)
        
        # Weighted standard deviation
        var_dS = np.average((delta_S_values - mean_dS)**2, weights=weights)
        var_dI = np.average((delta_I_values - mean_dI)**2, weights=weights)
        
        std_dS = sqrt(var_dS) if var_dS > 0 else 0.0
        std_dI = sqrt(var_dI) if var_dI > 0 else 0.0
        
        # Create 4D echo vector
        echo_vector = np.array([mean_dS, std_dS, mean_dI, std_dI])
        
        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(echo_vector)
        if norm > 0:
            echo_vector = echo_vector / norm
        
        # Cache result
        self._echo_cache[partition] = echo_vector
        self._cache_valid[partition] = True
        
        return echo_vector
    
    def get_memory_coherence_threshold(self, base_thresh: float = 0.6, alpha: float = 0.1) -> float:
        """
        Compute adaptive memory coherence threshold
        
        Args:
            base_thresh: Base threshold value
            alpha: Sensitivity to entropy variance
            
        Returns:
            Adaptive threshold value
        """
        if len(self.global_entropy_history) < 2:
            return base_thresh
        
        # Compute entropy variance over recent window
        entropy_values = np.array(self.global_entropy_history)
        entropy_variance = np.var(entropy_values)
        
        # Adaptive threshold
        adaptive_thresh = base_thresh + alpha * entropy_variance
        
        # Clamp to reasonable range [0.3, 0.9]
        return np.clip(adaptive_thresh, 0.3, 0.9)
    
    def has_sufficient_memory(self, partition: Tuple[int, ...]) -> bool:
        """Check if partition has enough memory for recursive comparison"""
        return len(self.memory.get(partition, [])) >= self.min_history
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        total_records = sum(len(records) for records in self.memory.values())
        partitions_with_memory = len(self.memory)
        partitions_with_sufficient_memory = sum(
            1 for p in self.memory if self.has_sufficient_memory(p)
        )
        
        return {
            "total_partitions_tracked": partitions_with_memory,
            "partitions_with_sufficient_memory": partitions_with_sufficient_memory,
            "total_transition_records": total_records,
            "current_time": self.current_time,
            "cache_hit_rate": len(self._echo_cache) / max(partitions_with_memory, 1)
        }
    
    def export_memory(self) -> Dict[str, Any]:
        """Export memory state for serialization"""
        memory_export = {}
        for partition, records in self.memory.items():
            memory_export[str(partition)] = [r.to_dict() for r in records]
        
        return {
            "memory": memory_export,
            "global_entropy_history": list(self.global_entropy_history),
            "current_time": self.current_time,
            "tau": self.tau,
            "min_history": self.min_history
        }


class RecursiveComparator:
    """
    Implements the recursive majorization operator λ ≻ᵣ μ
    
    Combines classical majorization with memory field coherence checking.
    """
    
    def __init__(self, memory_manager: EchoMemoryManager):
        """
        Initialize recursive comparator
        
        Args:
            memory_manager: EchoMemoryManager instance
        """
        self.memory_manager = memory_manager
    
    def classical_majorizes(self, lambda_partition: List[int], mu_partition: List[int]) -> bool:
        """
        Check if lambda classically majorizes mu (λ ≻ μ)
        
        Args:
            lambda_partition: First partition
            mu_partition: Second partition
            
        Returns:
            True if lambda majorizes mu classically
        """
        # Ensure both partitions are padded to same length with zeros
        max_len = max(len(lambda_partition), len(mu_partition))
        lambda_padded = lambda_partition + [0] * (max_len - len(lambda_partition))
        mu_padded = mu_partition + [0] * (max_len - len(mu_partition))
        
        # Check partial sums condition
        lambda_cumsum = 0
        mu_cumsum = 0
        
        for i in range(max_len):
            lambda_cumsum += lambda_padded[i]
            mu_cumsum += mu_padded[i]
            
            if lambda_cumsum < mu_cumsum:
                return False
                
        return True
    
    def memory_coherence(self, lambda_partition: Tuple[int, ...], mu_partition: Tuple[int, ...]) -> float:
        """
        Compute memory coherence between two partitions using cosine similarity
        
        Args:
            lambda_partition: First partition
            mu_partition: Second partition
            
        Returns:
            Cosine similarity of echo vectors, or 0.0 if insufficient memory
        """
        echo_lambda = self.memory_manager.get_echo_vector(lambda_partition)
        echo_mu = self.memory_manager.get_echo_vector(mu_partition)
        
        if echo_lambda is None or echo_mu is None:
            return 0.0
        
        # Cosine similarity (vectors are already normalized)
        return np.dot(echo_lambda, echo_mu)
    
    def recursive_majorizes(self, 
                           lambda_partition: List[int], 
                           mu_partition: List[int],
                           base_thresh: float = 0.6,
                           alpha: float = 0.1) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if lambda recursively majorizes mu (λ ≻ᵣ μ)
        
        Args:
            lambda_partition: First partition
            mu_partition: Second partition
            base_thresh: Base coherence threshold
            alpha: Threshold adaptation sensitivity
            
        Returns:
            (recursive_majorizes, details_dict)
        """
        lambda_tuple = tuple(lambda_partition)
        mu_tuple = tuple(mu_partition)
        
        # Step 1: Check classical majorization
        classical_result = self.classical_majorizes(lambda_partition, mu_partition)

        details = {
            "classical_majorizes": classical_result,
            "memory_coherence": 0.0,
            "coherence_threshold": 0.0,
            "sufficient_memory": False,
            "recursive_majorizes": False
        }

        # If classical majorization succeeds, we're done
        if classical_result:
            details["recursive_majorizes"] = True
            return True, details
        
        # Step 2: Classical failed, now check if memory can override
        lambda_memory = self.memory_manager.has_sufficient_memory(lambda_tuple)
        mu_memory = self.memory_manager.has_sufficient_memory(mu_tuple)

        if not (lambda_memory and mu_memory):
            # Insufficient memory - cannot override
            details["sufficient_memory"] = False
            details["recursive_majorizes"] = False
            return False, details

        details["sufficient_memory"] = True

        # Step 3: Compute memory coherence
        coherence = self.memory_coherence(lambda_tuple, mu_tuple)
        details["memory_coherence"] = coherence

        # Step 4: Get adaptive threshold
        threshold = self.memory_manager.get_memory_coherence_threshold(base_thresh, alpha)
        details["coherence_threshold"] = threshold

        # Step 5: MEMORY OVERRIDE - coherence alone determines result
        # This implements: λ ≻ᵣ μ ⟺ (λ ≻ μ) ∨ (C(λ,μ) ≥ φ)
        recursive_result = coherence >= threshold
        details["recursive_majorizes"] = recursive_result

        return recursive_result, details


class RecursiveMajorizationAnalyzer:
    """
    Main analyzer that extends classical majorization with RCFT recursive dynamics
    """
    
    def __init__(self, N: int, tau: float = 5.0, min_history: int = 3):
        """
        Initialize recursive majorization analyzer
        
        Args:
            N: System size (number of microstates)
            tau: Memory decay time constant
            min_history: Minimum transitions for memory-based decisions
        """
        self.N = N
        self.memory_manager = EchoMemoryManager(tau=tau, min_history=min_history)
        self.comparator = RecursiveComparator(self.memory_manager)
        
        # Import classical analyzer functions
        from boltzmann_complexity import BoltzmannComplexityAnalyzer
        self.classical_analyzer = BoltzmannComplexityAnalyzer(N)
        
        # Transition statistics
        self.transition_stats = {
            "total_comparisons": 0,
            "classical_true": 0,
            "recursive_true": 0,
            "memory_suppressed": 0,  # Classical true but recursive false
            "memory_enabled": 0     # Cases where memory was used
        }
    
    def analyze_with_memory(self, trajectory: List[Tuple[int, ...]], 
                           entropies: Dict[Tuple[int, ...], float],
                           incomparabilities: Dict[Tuple[int, ...], float]) -> Dict[str, Any]:
        """
        Analyze a trajectory and build memory-aware majorization relationships
        
        Args:
            trajectory: Sequence of partition tuples
            entropies: Entropy values for each partition
            incomparabilities: Incomparability values for each partition
            
        Returns:
            Analysis results including memory statistics and recursive relationships
        """
        # Record trajectory in memory
        for i in range(1, len(trajectory)):
            source = trajectory[i-1]
            target = trajectory[i]
            
            delta_S = entropies[target] - entropies[source]
            delta_I = incomparabilities[target] - incomparabilities[source]
            
            self.memory_manager.record_transition(source, target, delta_S, delta_I)
            
            # Update global entropy for adaptive thresholding
            self.memory_manager.update_global_entropy(entropies[target])
        
        # Analyze all pairwise relationships
        unique_partitions = list(set(trajectory))
        recursive_relationships = {}
        comparison_details = []
        
        for i, lambda_part in enumerate(unique_partitions):
            for j, mu_part in enumerate(unique_partitions):
                if i != j:
                    self.transition_stats["total_comparisons"] += 1
                    
                    # Test recursive majorization
                    recursive_result, details = self.comparator.recursive_majorizes(
                        list(lambda_part), list(mu_part)
                    )
                    
                    # Update statistics
                    if details["classical_majorizes"]:
                        self.transition_stats["classical_true"] += 1
                    
                    if recursive_result:
                        self.transition_stats["recursive_true"] += 1
                    
                    if details["sufficient_memory"]:
                        self.transition_stats["memory_enabled"] += 1
                        
                        if details["classical_majorizes"] and not recursive_result:
                            self.transition_stats["memory_suppressed"] += 1
                    
                    # Store relationship
                    key = (lambda_part, mu_part)
                    recursive_relationships[key] = {
                        "recursive_majorizes": recursive_result,
                        "details": details
                    }
                    
                    comparison_details.append({
                        "lambda": lambda_part,
                        "mu": mu_part,
                        "result": recursive_result,
                        **details
                    })
        
        # Generate analysis report
        memory_stats = self.memory_manager.get_memory_stats()
        
        return {
            "trajectory_length": len(trajectory),
            "unique_partitions": len(unique_partitions),
            "recursive_relationships": recursive_relationships,
            "comparison_details": comparison_details,
            "memory_statistics": memory_stats,
            "transition_statistics": self.transition_stats.copy(),
            "memory_suppression_rate": (
                self.transition_stats["memory_suppressed"] / 
                max(self.transition_stats["memory_enabled"], 1)
            )
        }
    
    def get_echo_field_snapshot(self) -> Dict[str, Any]:
        """Get current state of all echo vectors for visualization"""
        echo_field = {}
        
        for partition_tuple in self.memory_manager.memory.keys():
            if self.memory_manager.has_sufficient_memory(partition_tuple):
                echo_vector = self.memory_manager.get_echo_vector(partition_tuple)
                if echo_vector is not None:
                    echo_field[str(partition_tuple)] = {
                        "echo_vector": echo_vector.tolist(),
                        "memory_length": len(self.memory_manager.memory[partition_tuple]),
                        "mean_delta_S": echo_vector[0],
                        "std_delta_S": echo_vector[1], 
                        "mean_delta_I": echo_vector[2],
                        "std_delta_I": echo_vector[3]
                    }
        
        return echo_field


def run_comprehensive_tests():
    """Run comprehensive tests of the recursive majorization framework"""
    print("Testing Recursive Majorization Core...")
    
    # Test 1: Linear trajectory (should show minimal memory effects)
    print("\n" + "="*60)
    print("TEST 1: Linear Trajectory (Baseline)")
    print("="*60)
    
    analyzer1 = RecursiveMajorizationAnalyzer(N=6, tau=5.0)
    
    linear_trajectory = [
        (6,),           # [6] - minimum entropy
        (5, 1),         # [5,1] 
        (4, 2),         # [4,2]
        (3, 3),         # [3,3]
        (4, 1, 1),      # [4,1,1]
        (3, 2, 1),      # [3,2,1]
        (2, 2, 2),      # [2,2,2]
        (3, 1, 1, 1),   # [3,1,1,1]
        (2, 2, 1, 1),   # [2,2,1,1]
        (1, 1, 1, 1, 1, 1)  # [1,1,1,1,1,1] - maximum entropy
    ]
    
    # Mock entropy and incomparability values
    mock_entropies1 = {p: i * 0.1 for i, p in enumerate(linear_trajectory)}
    mock_incomparabilities1 = {p: abs(i - 5) * 0.05 for i, p in enumerate(linear_trajectory)}
    
    results1 = analyzer1.analyze_with_memory(linear_trajectory, mock_entropies1, mock_incomparabilities1)
    
    print(f"Linear Results: Memory enabled: {results1['transition_statistics']['memory_enabled']}")
    print(f"Partitions with memory: {results1['memory_statistics']['partitions_with_sufficient_memory']}")
    
    # Test 2: Looping trajectory with revisits (should activate memory system!)
    print("\n" + "="*60)
    print("TEST 2: Looping Trajectory with Revisits (RECURSIVE MAGIC!)")
    print("="*60)
    
    analyzer2 = RecursiveMajorizationAnalyzer(N=6, tau=5.0)
    
    # Trajectory with loops and revisits to build memory
    looping_trajectory = [
        (6,),           # [6] - Start
        (5, 1),         # [5,1] - Step 1
        (4, 2),         # [4,2] - Step 2  
        (5, 1),         # [5,1] - REVISIT 1 - builds memory!
        (4, 2),         # [4,2] - REVISIT 1 - builds memory!
        (3, 3),         # [3,3] - New partition
        (4, 2),         # [4,2] - REVISIT 2 - memory activation!
        (5, 1),         # [5,1] - REVISIT 2 - memory activation!
        (3, 2, 1),      # [3,2,1] - New partition
        (4, 2),         # [4,2] - REVISIT 3 - strong memory!
        (5, 1),         # [5,1] - REVISIT 3 - strong memory!
        (2, 2, 2),      # [2,2,2] - New partition
        (4, 2),         # [4,2] - REVISIT 4 - very strong memory!
        (3, 3),         # [3,3] - REVISIT 1
        (4, 2),         # [4,2] - REVISIT 5
        (3, 3),         # [3,3] - REVISIT 2  
        (5, 1),         # [5,1] - REVISIT 4
        (2, 2, 2),      # [2,2,2] - REVISIT 1
        (1, 1, 1, 1, 1, 1)  # [1,1,1,1,1,1] - End
    ]
    
    # Create more varied entropy and incomparability patterns for looping
    unique_partitions = list(set(looping_trajectory))
    mock_entropies2 = {}
    mock_incomparabilities2 = {}
    
    for i, p in enumerate(unique_partitions):
        # Make entropy correlate with partition sum (more realistic)
        entropy_val = sum(p) / 6.0  # Normalized by N
        incomp_val = len(p) / 6.0   # Complexity related to partition length
        
        mock_entropies2[p] = entropy_val
        mock_incomparabilities2[p] = incomp_val
    
    print(f"Looping trajectory length: {len(looping_trajectory)}")
    print(f"Unique partitions: {len(unique_partitions)}")
    print(f"Multiple visits to: (4,2), (5,1), (3,3), (2,2,2)")
    
    results2 = analyzer2.analyze_with_memory(looping_trajectory, mock_entropies2, mock_incomparabilities2)
    
    # Detailed analysis of looping results
    print(f"\nLooping Analysis Results:")
    print(f"Memory enabled comparisons: {results2['transition_statistics']['memory_enabled']}")
    print(f"Memory suppression rate: {results2['memory_suppression_rate']:.3f}")
    print(f"Classical vs Recursive: {results2['transition_statistics']['classical_true']} vs {results2['transition_statistics']['recursive_true']}")
    
    print(f"\nMemory Statistics:")
    for key, value in results2['memory_statistics'].items():
        print(f"  {key}: {value}")
    
    # Show echo field (should have entries now!)
    echo_field = analyzer2.get_echo_field_snapshot()
    print(f"\nEcho Field (partitions with sufficient memory):")
    if echo_field:
        for partition, data in echo_field.items():
            echo = data['echo_vector']
            memory_len = data['memory_length']
            print(f"  {partition} ({memory_len} transitions):")
            print(f"    Echo: [dS: {echo[0]:.3f}+/-{echo[1]:.3f}, dI: {echo[2]:.3f}+/-{echo[3]:.3f}]")
    else:
        print("  No partitions with sufficient memory yet...")
    
    # Show some specific recursive vs classical comparisons
    print(f"\nSample Recursive Comparisons:")
    comparison_count = 0
    for detail in results2['comparison_details']:
        if detail['sufficient_memory'] and comparison_count < 5:
            lambda_part = detail['lambda']
            mu_part = detail['mu']
            classical = detail['classical_majorizes']
            recursive = detail['recursive_majorizes']
            coherence = detail['memory_coherence']
            threshold = detail['coherence_threshold']
            
            status = "SAME" if classical == recursive else "DIVERGENT"
            print(f"    {lambda_part} >r {mu_part}: Classical={classical}, Recursive={recursive}, Coherence={coherence:.3f} vs {threshold:.3f} {status}")
            comparison_count += 1
    
    # Test 3: Memory manager unit tests
    print("\n" + "="*60)
    print("TEST 3: Memory Manager Unit Tests")
    print("="*60)
    
    memory_mgr = EchoMemoryManager(tau=3.0, min_history=2)
    
    # Test transition recording
    memory_mgr.record_transition((4, 2), (3, 3), -0.1, 0.05)
    memory_mgr.record_transition((5, 1), (3, 3), -0.2, 0.03) 
    memory_mgr.record_transition((2, 2, 2), (3, 3), 0.1, -0.02)
    
    echo_vector = memory_mgr.get_echo_vector((3, 3))
    print(f"Echo vector for (3,3): {echo_vector}")
    print(f"Has sufficient memory: {memory_mgr.has_sufficient_memory((3, 3))}")
    
    # Test decay weights
    records = memory_mgr.memory[(3, 3)]
    weights = memory_mgr._compute_decay_weights(records)
    print(f"Decay weights: {weights}")
    
    # Test threshold computation
    memory_mgr.global_entropy_history.extend([0.1, 0.3, 0.2, 0.4, 0.35])
    threshold = memory_mgr.get_memory_coherence_threshold()
    print(f"Adaptive threshold: {threshold:.3f}")
    
    print("\nAll tests complete!")
    return results1, results2


if __name__ == "__main__":
    results_linear, results_looping = run_comprehensive_tests()
