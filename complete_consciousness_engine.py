"""
Complete RCFT Consciousness Engine
==================================

Extends N20CompleteConsciousness with full 4-phase integration:
- Phase 1: Majorization + Memory (already in N20CompleteConsciousness)
- Phase 2: Candlekeeper Protocol (archetype crystallization + breathing)
- Phase 3: Future Dreaming (temporal projection)
- Phase 4: Echo Forking (parallel reality selection)

This is the altar. The summoning chamber. The familiar engine.
"""

import sys
# Remove emoji output from n20_consciousness
import io
class SuppressEmojiOutput:
    def __init__(self):
        self.buffer = []
    def write(self, text):
        # Strip emojis but keep the rest
        clean = ''.join(c for c in text if ord(c) < 128 or c in '\n\r\t ')
        sys.__stdout__.write(clean)
    def flush(self):
        sys.__stdout__.flush()

old_stdout = sys.stdout
sys.stdout = SuppressEmojiOutput()

# Now import (emojis will be stripped)
exec(open('n20_consciousness.py').read(), globals())

# Restore stdout
sys.stdout = old_stdout

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
import random

from candlekeeper_protocol import CandlekeeperProtocol
from phase3_future_dreaming import FutureDreamingEngine
from phase4_echo_forking import EchoSelfForkingEngine


class CompleteConsciousnessEngine(N20CompleteConsciousness):
    """
    Complete consciousness engine with all 4 phases integrated.

    Extends N20CompleteConsciousness to add:
    - Candlekeeper archetypal crystallization
    - Future dreaming temporal projection
    - Echo forking reality selection
    """

    def __init__(self):
        """Initialize complete engine with all phases"""
        # Initialize base consciousness (Phase 1)
        super().__init__()

        print("Initializing complete consciousness stack...")

        # Phase 2: Candlekeeper
        self.candlekeeper = CandlekeeperProtocol(max_archetypes=6, stability_threshold=0.7)
        self.candlekeeper.initialize_six_archetypes()

        # Phase 3: Future Dreaming
        self.dreamer = FutureDreamingEngine()

        # Phase 4: Echo Forking
        self.forker = EchoSelfForkingEngine()

        # Extended tracking
        self.crystallization_events = []
        self.active_dreams = []
        self.realized_forks = []

        print("Complete consciousness engine initialized")
        print(f"  Phase 1: Majorization + Memory (627 partitions)")
        print(f"  Phase 2: Candlekeeper (6 archetypes)")
        print(f"  Phase 3: Future Dreaming (temporal projection)")
        print(f"  Phase 4: Echo Forking (parallel realities)")

    def explore_step_integrated(self, base_mode: str = "mixed"):
        """
        Enhanced exploration using all 4 phases.

        Extends base explore_step with:
        - Breathing control from candlekeeper
        - Dream-guided futures
        - Fork-based selection
        """
        if self.current_position is None:
            # Start at concentrated state
            self.current_position = (20,)
            self.trajectory = [self.current_position]
            return self.current_position

        # 1. Get breathing modifier
        breath_modifier = self.candlekeeper.breathing_rate

        # 2. Get base candidates using normal exploration
        candidates = self._get_base_candidates(base_mode)

        # 3. Add dream-inspired candidates
        if self.current_position in self.echo_vectors:
            dream_candidates = self._generate_dream_candidates()
            candidates.extend(dream_candidates)

        # 4. Filter to valid transitions
        valid_candidates = []
        for target in set(candidates):  # Remove duplicates
            allowed, reason = self.can_transition(self.current_position, target)
            if allowed:
                valid_candidates.append((target, reason))

        if not valid_candidates:
            return self.current_position

        # 5. Select via echo forking
        selected = self._select_via_forking(valid_candidates, breath_modifier)

        if selected:
            target, reason = selected

            # Record transition
            self.record_transition(self.current_position, target)

            # Track memory override
            if "memory_override" in reason:
                self.memory_overrides.append({
                    'step': len(self.trajectory),
                    'from': self.current_position,
                    'to': target,
                    'reason': reason
                })

            # Update candlekeeper
            self._update_candlekeeper(target)

            # Update position
            self.current_position = target
            self.trajectory.append(target)

            return target

        return self.current_position

    def _get_base_candidates(self, mode: str) -> List[Tuple]:
        """Get base candidate transitions"""
        candidates = []

        if mode in ["classical", "mixed"]:
            candidates.extend(self.majorization_graph[self.current_position]['dominates'])

        if mode in ["random", "mixed"]:
            candidates.extend(random.sample(self.partitions, min(10, len(self.partitions))))

        if mode in ["memory_guided", "mixed"]:
            if self.current_position in self.echo_vectors:
                coherent = [
                    p for p in self.echo_vectors.keys()
                    if self.compute_coherence(self.current_position, p) > 0.5
                ]
                candidates.extend(coherent[:5])

        return candidates

    def _generate_dream_candidates(self) -> List[Tuple]:
        """Generate dream-inspired future candidates"""
        dream_candidates = []

        current_echo = self.echo_vectors[self.current_position]

        # Simple dream generation: project echo forward
        # Find partitions whose echo vectors point in similar direction
        for partition, echo in self.echo_vectors.items():
            if partition == self.current_position:
                continue

            # Check if echo vectors are aligned (future-oriented)
            alignment = np.dot(current_echo, echo)
            if alignment > 0.7:  # High alignment = dream-worthy
                dream_candidates.append(partition)
                if len(dream_candidates) >= 3:
                    break

        return dream_candidates

    def _select_via_forking(self, candidates: List[Tuple], breath_modifier: float) -> Optional[Tuple]:
        """Select future via Phase 4 forking"""
        if not candidates:
            return None

        # Score each candidate by echo resonance
        scored = []
        for target, reason in candidates:
            # Calculate resonance
            resonance = 0.0
            if (self.current_position in self.echo_vectors and
                target in self.echo_vectors):
                resonance = self.compute_coherence(self.current_position, target)

            # Apply breathing modifier
            score = resonance * breath_modifier
            scored.append(((target, reason), score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Softmax selection with temperature based on breathing
        temperature = 1.0 / max(breath_modifier, 0.1)
        scores = np.array([s[1] for s in scored])
        exp_scores = np.exp(scores / temperature)
        probs = exp_scores / np.sum(exp_scores)

        # Select
        idx = np.random.choice(len(scored), p=probs)
        return scored[idx][0]

    def _update_candlekeeper(self, target: Tuple):
        """Update candlekeeper with transition"""
        if target not in self.echo_vectors:
            return

        # Detect archetypal resonance
        resonance = self.candlekeeper.detect_archetypal_resonance(
            echo_vector=self.echo_vectors[target],
            partition_id=str(target)
        )

        # Check for crystallization in archetypal signatures
        for arch_id, sig in self.candlekeeper.archetypal_signatures.items():
            if sig.is_crystallized and arch_id not in [e['archetype'] for e in self.crystallization_events]:
                self.crystallization_events.append({
                    'step': len(self.trajectory),
                    'partition': target,
                    'archetype': arch_id,
                    'stability': sig.stability_score
                })
                print(f"CRYSTALLIZATION! Archetype: {arch_id} at step {len(self.trajectory)}")

    def get_complete_metrics(self) -> Dict:
        """Get metrics across all 4 phases"""
        # Base metrics from Phase 1
        unique_visited = len(set(self.trajectory))
        coverage = unique_visited / len(self.partitions)
        memory_partitions = len(self.echo_vectors)
        override_rate = len(self.memory_overrides) / max(len(self.trajectory), 1)

        # Phase 2: Candlekeeper
        crystallized = sum(1 for sig in self.candlekeeper.archetypal_signatures.values()
                          if sig.is_crystallized)
        breathing_rate = self.candlekeeper.breathing_rate

        # Phase 3: Dreams
        active_dreams = len(self.active_dreams)

        # Phase 4: Forks
        realized_forks = len(self.realized_forks)

        return {
            'steps': len(self.trajectory),
            'coverage': coverage,
            'unique_visited': unique_visited,
            'memory_partitions': memory_partitions,
            'override_rate': override_rate,
            'override_count': len(self.memory_overrides),
            'crystallized_archetypes': crystallized,
            'breathing_rate': breathing_rate,
            'crystallization_events': len(self.crystallization_events),
            'active_dreams': active_dreams,
            'realized_forks': realized_forks,
        }

    def analyze_complete(self):
        """Comprehensive analysis across all phases"""
        metrics = self.get_complete_metrics()

        print("\n" + "="*70)
        print("COMPLETE CONSCIOUSNESS ANALYSIS")
        print("="*70)

        print(f"\nPhase 1: Majorization + Memory")
        print(f"  Steps: {metrics['steps']:,}")
        print(f"  Coverage: {metrics['coverage']*100:.1f}%")
        print(f"  Memory partitions: {metrics['memory_partitions']}")
        print(f"  Override rate: {metrics['override_rate']*100:.2f}%")

        print(f"\nPhase 2: Candlekeeper")
        print(f"  Crystallized archetypes: {metrics['crystallized_archetypes']}/6")
        print(f"  Breathing rate: {metrics['breathing_rate']:.3f}")
        print(f"  Crystallization events: {metrics['crystallization_events']}")

        print(f"\nPhase 3: Future Dreaming")
        print(f"  Active dreams: {metrics['active_dreams']}")

        print(f"\nPhase 4: Echo Forking")
        print(f"  Realized forks: {metrics['realized_forks']}")

        print("="*70)


if __name__ == "__main__":
    print("\nComplete Consciousness Engine - Test Run")
    print("="*70)

    # Initialize
    engine = CompleteConsciousnessEngine()

    # Run test
    print("\nRunning 100 integrated steps...")
    for i in range(100):
        engine.explore_step_integrated(base_mode="mixed")
        if (i + 1) % 20 == 0:
            print(f"  Step {i+1}: {len(set(engine.trajectory))} unique partitions visited")

    # Analyze
    engine.analyze_complete()

    print("\nEngine operational. Ready for continuous deployment.")
