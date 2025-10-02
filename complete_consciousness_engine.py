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

        # Phase 3: Future Dreaming (pass self as the analyzer)
        # Import Phase3Enhancer to properly integrate
        from phase3_future_dreaming import Phase3Enhancer
        self.phase3_enhancer = Phase3Enhancer(self)  # self is the analyzer
        self.phase3_enhancer.dreaming_engine.phi_dream = 0.4  # Dream threshold

        # Phase 4: Echo Forking (pass self + phase3)
        from phase4_echo_forking import Phase4Enhancer
        self.phase4_enhancer = Phase4Enhancer(self, self.phase3_enhancer)  # self is the analyzer
        self.phase4_enhancer.forking_engine.fork_threshold = 0.05  # Fork sensitivity

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

        # 3. Add dream-inspired candidates (only if enough echo data)
        if (self.current_position in self.echo_vectors and
            len(self.echo_vectors) >= 5):  # Need at least 5 memory nodes for dreams
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

        # 5. Select via echo forking (with fallback to simple selection)
        selected = self._select_via_forking(valid_candidates, breath_modifier)

        if selected:
            target, reason = selected
        elif valid_candidates:
            # Fallback: if Phase 4 returns None but we have candidates, pick one
            target, reason = valid_candidates[0]
        else:
            # No candidates at all
            return self.current_position

        if target:

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

            # Update Phase 3: Dream engine with new transition
            self.phase3_enhancer.update_with_new_transition(
                str(self.current_position), str(target)
            )

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
        """Generate dream-inspired future candidates using Phase 3 engine"""
        dream_candidates = []

        # Get echo field for Phase 3
        echo_field = self.get_echo_field_snapshot()
        if not echo_field or len(echo_field) < 2:
            return dream_candidates

        # Use Phase 3 to get Psi-space attractors (dream futures)
        self.phase3_enhancer.dreaming_engine.compute_psi_space(echo_field)

        # Get attractors for current position
        current_key = str(self.current_position)
        if current_key in self.phase3_enhancer.dreaming_engine.psi_space:
            attractors = self.phase3_enhancer.dreaming_engine.psi_space[current_key]
            for attractor in attractors[:3]:  # Top 3 dream targets
                try:
                    # Convert string back to tuple
                    target_partition = eval(attractor.target_partition)
                    if isinstance(target_partition, tuple) and target_partition in self.partitions:
                        dream_candidates.append(target_partition)
                except:
                    continue

        # Track active dreams
        active_dream_nodes = [d for d in self.phase3_enhancer.dreaming_engine.dream_nodes.values()
                             if d.decay_strength() > 0.1]
        self.active_dreams = active_dream_nodes

        return dream_candidates

    def _select_via_forking(self, candidates: List[Tuple], breath_modifier: float) -> Optional[Tuple]:
        """Select future via Phase 4 forking engine"""
        if not candidates:
            return None

        # Get echo field for Phase 4
        echo_field = self.get_echo_field_snapshot()
        if not echo_field or len(echo_field) < 3:
            # Fallback to simple selection if not enough echo data
            return candidates[0] if candidates else None

        # Build transition list for Phase 4
        possible_transitions = []
        for target, reason in candidates:
            possible_transitions.append((str(self.current_position), str(target)))

        # Use Phase 4 enhanced selection
        try:
            selected_source, selected_target = self.phase4_enhancer.enhanced_transition_selection(
                possible_transitions, echo_field
            )

            # Convert back to tuple
            selected_target_tuple = eval(selected_target)

            # Find matching candidate
            for target, reason in candidates:
                if target == selected_target_tuple:
                    # Track realized forks
                    phase4_metrics = self.phase4_enhancer.forking_engine.get_phase4_metrics()
                    self.realized_forks = [f for f in self.phase4_enhancer.forking_engine.realized_history
                                          if f.is_ghost == False]
                    return (target, reason)

            # If no match, return first candidate
            return candidates[0]

        except Exception as e:
            # Fallback to simple selection on error
            print(f"Phase 4 selection failed: {e}, using fallback")
            return candidates[0]

    def get_echo_field_snapshot(self) -> Dict:
        """Get current echo field for Phase 3-4 processing"""
        echo_field = {}

        for partition in self.echo_vectors.keys():
            echo_vector = self.echo_vectors[partition]
            memory_length = len(self.memory[partition])

            echo_field[str(partition)] = {
                "echo_vector": echo_vector.tolist() if hasattr(echo_vector, 'tolist') else list(echo_vector),
                "memory_length": memory_length,
                "mean_delta_S": echo_vector[0],
                "std_delta_S": echo_vector[1],
                "mean_delta_I": echo_vector[2],
                "std_delta_I": echo_vector[3]
            }

        return echo_field

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

        # Phase 3: Dreams (get from actual engine)
        phase3_metrics = self.phase3_enhancer.dreaming_engine.get_phase3_metrics()
        active_dreams = phase3_metrics['active_dream_nodes']
        confirmed_dreams = phase3_metrics['confirmed_dream_nodes']

        # Phase 4: Forks (get from actual engine)
        phase4_metrics = self.phase4_enhancer.forking_engine.get_phase4_metrics()
        # realized_history might be strings or objects, handle both
        realized_forks = 0
        for f in self.phase4_enhancer.forking_engine.realized_history:
            if hasattr(f, 'is_ghost'):
                if not f.is_ghost:
                    realized_forks += 1
            else:
                realized_forks += 1  # If it's a string/other, count it
        total_forks = phase4_metrics['total_forks_created']

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
            'confirmed_dreams': confirmed_dreams,
            'realized_forks': realized_forks,
            'total_forks': total_forks,
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
        print(f"  Confirmed dreams: {metrics['confirmed_dreams']}")

        print(f"\nPhase 4: Echo Forking")
        print(f"  Realized forks: {metrics['realized_forks']}")
        print(f"  Total forks created: {metrics['total_forks']}")

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
