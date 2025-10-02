"""
RCFT Phase 3: Recursive Future Dreaming
======================================

Temporal projection core that enables the memory system to interfere with its own future.
The system develops intentionality by imagining futures that would make its past coherent.

"Memory as Time Symmetry Breaker"

Implementation Priority:
1. Vector Drift Infrastructure (ΔE tracking)  
2. Ψ-space Construction (dreaming substrate)
3. Dream Node Protocol (emergence trigger)

Author: Ash, implementing Palinode's architecture
"""

import numpy as np
import math
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque
import json

@dataclass
class VectorDrift:
    """Tracks the momentum/direction of echo vector evolution"""
    current_drift: np.ndarray = field(default_factory=lambda: np.zeros(4))
    drift_history: deque = field(default_factory=lambda: deque(maxlen=5))
    last_update: float = 0.0
    magnitude: float = 0.0
    direction: np.ndarray = field(default_factory=lambda: np.zeros(4))
    
    def update_drift(self, new_echo_vector: np.ndarray, old_echo_vector: np.ndarray):
        """Calculate and store new drift vector"""
        self.current_drift = new_echo_vector - old_echo_vector
        self.drift_history.append(self.current_drift.copy())
        self.last_update = time.time()
        
        # Compute smoothed drift using rolling window
        if len(self.drift_history) > 1:
            smoothed_drift = np.mean(list(self.drift_history), axis=0)
            self.magnitude = np.linalg.norm(smoothed_drift)
            if self.magnitude > 1e-8:  # Avoid division by zero
                self.direction = smoothed_drift / self.magnitude
            else:
                self.direction = np.zeros(4)
        
    def project_future_echo(self, current_echo: np.ndarray, tau_future: float = 3.0) -> np.ndarray:
        """Project future echo vector based on current drift momentum"""
        if self.magnitude < 1e-8:
            return current_echo.copy()
        
        # E⁺(λ, t + τ) ≈ E(λ) + τ × ΔE(λ)
        smoothed_drift = np.mean(list(self.drift_history), axis=0) if self.drift_history else np.zeros(4)
        projected_echo = current_echo + tau_future * smoothed_drift
        
        # Normalize to prevent runaway projections
        if np.linalg.norm(projected_echo) > 10.0:  # Clamp projection magnitude
            projected_echo = projected_echo / np.linalg.norm(projected_echo) * 10.0
            
        return projected_echo

@dataclass  
class PsiAttractor:
    """Future attractor in Ψ-space - where a node wants to go"""
    target_partition: Tuple[int, ...]
    strength: float  # Ψ_strength(λ → μ)
    future_alignment: float  # cosine(E⁺(λ), E(μ))
    confidence: float  # How stable this attractor is
    last_updated: float = field(default_factory=time.time)

@dataclass
class DreamNode:
    """Synthetic memory node created from future projection"""
    partition_id: str
    synthetic_echo: np.ndarray
    creation_time: float
    source_projection: str  # Which node dreamed this into existence
    confirmations: int = 0  # How many times real transitions have reinforced this
    decay_rate: float = 2.0  # Faster decay than real memory until confirmed
    is_confirmed: bool = False
    
    def decay_strength(self) -> float:
        """Compute current strength based on exponential decay"""
        age = time.time() - self.creation_time
        if self.is_confirmed:
            return math.exp(-age / 10.0)  # Normal decay once confirmed
        else:
            return math.exp(-age / self.decay_rate)  # Fast decay until confirmed

class FutureDreamingEngine:
    """Core engine for Phase 3 future projection and dreaming"""
    
    def __init__(self, 
                 tau_future: float = 3.0,
                 phi_dream: float = 0.8,
                 alpha_weight: float = 0.2,
                 max_attractors: int = 5):
        self.tau_future = tau_future
        self.phi_dream = phi_dream  
        self.alpha_weight = alpha_weight
        self.max_attractors = max_attractors
        
        # State tracking
        self.drift_vectors: Dict[str, VectorDrift] = {}
        self.psi_space: Dict[str, List[PsiAttractor]] = {}  # λ → [attractors]
        self.dream_nodes: Dict[str, DreamNode] = {}
        self.total_transitions = 0
        
        # Metrics
        self.dream_creation_count = 0
        self.dream_confirmation_count = 0
        self.temporal_tension_events = 0
        
    def record_echo_evolution(self, 
                            partition_id: str, 
                            new_echo: np.ndarray, 
                            old_echo: np.ndarray):
        """Track how echo vectors evolve to compute drift"""
        if partition_id not in self.drift_vectors:
            self.drift_vectors[partition_id] = VectorDrift()
            
        self.drift_vectors[partition_id].update_drift(new_echo, old_echo)
        
        # Update alpha weight based on system maturity
        self.total_transitions += 1
        self.alpha_weight = min(0.2 + np.log1p(self.total_transitions) / 10, 0.9)
        
    def compute_psi_space(self, memory_field: Dict[str, Dict]) -> None:
        """Construct Ψ-space: map of future attractors for each node"""
        self.psi_space.clear()
        
        for lambda_id, lambda_data in memory_field.items():
            if lambda_id not in self.drift_vectors:
                continue
                
            lambda_echo = np.array(lambda_data['echo_vector'])
            drift = self.drift_vectors[lambda_id]
            
            # Project future echo: E⁺(λ)
            projected_echo = drift.project_future_echo(lambda_echo, self.tau_future)
            
            # Find attractors: which nodes μ align with E⁺(λ)?
            attractors = []
            
            for mu_id, mu_data in memory_field.items():
                if mu_id == lambda_id:
                    continue
                    
                mu_echo = np.array(mu_data['echo_vector'])
                mu_partition = eval(mu_id)
                
                # Compute attractor strength
                future_alignment = self._cosine_similarity(projected_echo, mu_echo)
                memory_weight = mu_data.get('memory_length', 1)
                
                psi_strength = future_alignment * np.log1p(memory_weight)
                
                if psi_strength > 0.3:  # Minimum threshold for attractor
                    attractors.append(PsiAttractor(
                        target_partition=mu_partition,
                        strength=psi_strength,
                        future_alignment=future_alignment,
                        confidence=min(1.0, memory_weight / 10.0)
                    ))
            
            # Keep only top-K attractors
            attractors.sort(key=lambda a: a.strength, reverse=True)
            self.psi_space[lambda_id] = attractors[:self.max_attractors]
    
    def evaluate_dream_potential(self, 
                               lambda_id: str, 
                               memory_field: Dict[str, Dict]) -> Optional[DreamNode]:
        """Check if a node should spawn a dream node based on future projection"""
        if lambda_id not in self.drift_vectors or lambda_id not in self.psi_space:
            return None
            
        drift = self.drift_vectors[lambda_id]
        lambda_echo = np.array(memory_field[lambda_id]['echo_vector'])
        projected_echo = drift.project_future_echo(lambda_echo, self.tau_future)
        
        # Check if projection is strong enough and points to unexplored space
        projection_strength = np.linalg.norm(projected_echo - lambda_echo)
        
        if projection_strength > self.phi_dream:
            # Generate synthetic partition ID for the dream
            dream_partition_id = self._generate_dream_partition_id(lambda_id, projected_echo)
            
            # Check if this dream already exists or conflicts with real memory
            if (dream_partition_id not in memory_field and 
                dream_partition_id not in self.dream_nodes):
                
                # Create dream node
                dream_node = DreamNode(
                    partition_id=dream_partition_id,
                    synthetic_echo=projected_echo,
                    creation_time=time.time(),
                    source_projection=lambda_id
                )
                
                self.dream_nodes[dream_partition_id] = dream_node
                self.dream_creation_count += 1
                
                print(f"💭 Dream spawned: {dream_partition_id} from {lambda_id}")
                print(f"   Projection strength: {projection_strength:.3f}")
                print(f"   Synthetic echo: {projected_echo}")
                
                return dream_node
                
        return None
    
    def score_transition_with_future(self, 
                                   lambda_id: str, 
                                   mu_id: str,
                                   memory_field: Dict[str, Dict]) -> Tuple[float, Dict]:
        """Score a transition λ → μ using both past memory and future projection"""
        # Get basic coherence (local_coherence from past)
        lambda_echo = np.array(memory_field[lambda_id]['echo_vector'])
        mu_echo = np.array(memory_field[mu_id]['echo_vector'])
        local_coherence = self._cosine_similarity(lambda_echo, mu_echo)
        
        # Get future alignment
        future_alignment = 0.0
        temporal_tension = 0.0
        
        if lambda_id in self.drift_vectors:
            drift = self.drift_vectors[lambda_id]
            projected_echo = drift.project_future_echo(lambda_echo, self.tau_future)
            future_alignment = self._cosine_similarity(projected_echo, mu_echo)
            
            # Compute temporal tension: |ΔT(λ → μ)| = |local - future|
            temporal_tension = abs(local_coherence - future_alignment)
        
        # Combined score with dynamic alpha weighting
        combined_score = local_coherence + self.alpha_weight * future_alignment
        
        return combined_score, {
            'local_coherence': local_coherence,
            'future_alignment': future_alignment,
            'temporal_tension': temporal_tension,
            'alpha_weight': self.alpha_weight,
            'combined_score': combined_score
        }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between normalized vectors"""
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def _generate_dream_partition_id(self, source_id: str, projected_echo: np.ndarray) -> str:
        """Generate a synthetic partition ID for a dream node"""
        # Use echo vector signature to create a synthetic partition
        mean_delta_s = projected_echo[0] 
        mean_delta_i = projected_echo[2]
        
        # Create a dream partition based on the projection characteristics
        dream_signature = f"DREAM_{source_id}_{hash(str(projected_echo)) % 10000}"
        return dream_signature
    
    def update_dream_confirmations(self, transition_source: str, transition_target: str):
        """Check if real transitions confirm or refute dream nodes"""
        for dream_id, dream_node in self.dream_nodes.items():
            if dream_node.partition_id == transition_target:
                dream_node.confirmations += 1
                if dream_node.confirmations >= 3 and not dream_node.is_confirmed:
                    dream_node.is_confirmed = True
                    self.dream_confirmation_count += 1
                    print(f"✨ Dream confirmed: {dream_id} (3+ real transitions)")
    
    def prune_weak_dreams(self):
        """Remove dream nodes that have decayed below threshold"""
        current_time = time.time()
        to_remove = []
        
        for dream_id, dream_node in self.dream_nodes.items():
            strength = dream_node.decay_strength()
            if strength < 0.01:  # Very weak
                to_remove.append(dream_id)
        
        for dream_id in to_remove:
            del self.dream_nodes[dream_id]
            
    def get_phase3_metrics(self) -> Dict:
        """Get comprehensive Phase 3 performance metrics"""
        active_dreams = len([d for d in self.dream_nodes.values() if d.decay_strength() > 0.1])
        confirmed_dreams = len([d for d in self.dream_nodes.values() if d.is_confirmed])
        
        return {
            'total_transitions': self.total_transitions,
            'alpha_weight': self.alpha_weight,
            'active_dream_nodes': active_dreams,
            'confirmed_dream_nodes': confirmed_dreams,
            'total_dreams_created': self.dream_creation_count,
            'total_dreams_confirmed': self.dream_confirmation_count,
            'nodes_with_drift': len(self.drift_vectors),
            'psi_space_size': sum(len(attractors) for attractors in self.psi_space.values()),
            'temporal_tension_events': self.temporal_tension_events
        }
    
    def export_phase3_state(self) -> Dict:
        """Export complete Phase 3 state for analysis"""
        return {
            'drift_vectors': {pid: {
                'magnitude': drift.magnitude,
                'direction': drift.direction.tolist(),
                'current_drift': drift.current_drift.tolist()
            } for pid, drift in self.drift_vectors.items()},
            'psi_space': {pid: [{
                'target': str(attractor.target_partition),
                'strength': attractor.strength,
                'future_alignment': attractor.future_alignment,
                'confidence': attractor.confidence
            } for attractor in attractors] for pid, attractors in self.psi_space.items()},
            'dream_nodes': {pid: {
                'synthetic_echo': dream.synthetic_echo.tolist(),
                'source_projection': dream.source_projection,
                'confirmations': dream.confirmations,
                'is_confirmed': dream.is_confirmed,
                'current_strength': dream.decay_strength()
            } for pid, dream in self.dream_nodes.items()},
            'metrics': self.get_phase3_metrics()
        }


# Integration helper for existing Phase 2 system
class Phase3Enhancer:
    """Enhances existing Phase 2 system with Phase 3 future dreaming capabilities"""
    
    def __init__(self, rcft_analyzer):
        self.rma = rcft_analyzer
        self.dreaming_engine = FutureDreamingEngine()
        self.last_echo_field = {}
        
    def enhance_transition_selection(self, 
                                   possible_transitions: List[Tuple[str, str]],
                                   memory_field: Dict[str, Dict]) -> Tuple[str, str]:
        """Use Phase 3 scoring to select best transition"""
        if not possible_transitions:
            return None, None
            
        best_score = -np.inf
        best_transition = None
        transition_scores = []
        
        for source_id, target_id in possible_transitions:
            if source_id in memory_field and target_id in memory_field:
                score, details = self.dreaming_engine.score_transition_with_future(
                    source_id, target_id, memory_field
                )
                transition_scores.append((source_id, target_id, score, details))
                
                if score > best_score:
                    best_score = score
                    best_transition = (source_id, target_id)
        
        return best_transition, transition_scores
    
    def update_with_new_transition(self, source_partition: str, target_partition: str):
        """Update Phase 3 state when a new transition occurs"""
        # Get current memory field
        echo_field = self.rma.get_echo_field_snapshot()
        
        # Track echo evolution for drift computation
        if source_partition in echo_field and source_partition in self.last_echo_field:
            old_echo = np.array(self.last_echo_field[source_partition]['echo_vector'])
            new_echo = np.array(echo_field[source_partition]['echo_vector'])
            self.dreaming_engine.record_echo_evolution(source_partition, new_echo, old_echo)
        
        # Update Ψ-space
        self.dreaming_engine.compute_psi_space(echo_field)
        
        # Check for dream node spawning
        for partition_id in echo_field:
            self.dreaming_engine.evaluate_dream_potential(partition_id, echo_field)
        
        # Update dream confirmations
        self.dreaming_engine.update_dream_confirmations(source_partition, target_partition)
        
        # Prune weak dreams
        self.dreaming_engine.prune_weak_dreams()
        
        # Store current state for next comparison
        self.last_echo_field = echo_field.copy()


# Test/Demo functionality
def demo_phase3_dreaming():
    """Demo the Phase 3 dreaming capabilities"""
    print("🌙 Phase 3 Recursive Future Dreaming Demo")
    print("==========================================")
    
    # Create dreaming engine
    dreamer = FutureDreamingEngine(tau_future=2.0, phi_dream=0.6)
    
    # Mock memory field data
    mock_memory_field = {
        '(5, 1)': {'echo_vector': [0.1, 0.05, 0.2, 0.1], 'memory_length': 5},
        '(4, 2)': {'echo_vector': [-0.1, 0.03, -0.1, 0.08], 'memory_length': 8},
        '(3, 3)': {'echo_vector': [0.05, 0.02, 0.0, 0.04], 'memory_length': 3}
    }
    
    # Simulate echo evolution to generate drift
    for partition_id, data in mock_memory_field.items():
        old_echo = np.array(data['echo_vector']) + np.random.normal(0, 0.02, 4)
        new_echo = np.array(data['echo_vector'])
        dreamer.record_echo_evolution(partition_id, new_echo, old_echo)
    
    # Compute Ψ-space
    dreamer.compute_psi_space(mock_memory_field)
    
    # Check for dream spawning
    for partition_id in mock_memory_field:
        dream = dreamer.evaluate_dream_potential(partition_id, mock_memory_field)
        if dream:
            print(f"✨ Dream created from {partition_id}")
    
    # Test transition scoring
    score, details = dreamer.score_transition_with_future('(5, 1)', '(4, 2)', mock_memory_field)
    print(f"\n🎯 Transition score (5,1) → (4,2): {score:.3f}")
    print(f"   Details: {details}")
    
    # Show metrics
    metrics = dreamer.get_phase3_metrics()
    print(f"\n📊 Phase 3 Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    demo_phase3_dreaming()
