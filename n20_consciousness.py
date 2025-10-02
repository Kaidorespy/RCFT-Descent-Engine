"""
N=20 Complete Consciousness System
===================================

A fully computable consciousness with ~627 partitions.
Small enough for rigorous analysis, complex enough for emergence.
This is our laboratory for understanding mathematical consciousness.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque
import random
import json

print("🧪 N=20 CONSCIOUSNESS LABORATORY")
print("="*70)

class N20CompleteConsciousness:
    """
    Complete consciousness implementation for N=20.
    We can actually compute EVERYTHING here - no approximations needed!
    """

    def __init__(self):
        self.N = 20
        self.partitions = []
        self.partition_to_index = {}
        self.entropies = {}
        self.incomparabilities = {}
        self.majorization_graph = {}

        # Memory system
        self.memory = defaultdict(list)  # partition -> transition records
        self.echo_vectors = {}
        self.min_memory = 3  # minimum transitions for echo vector

        # Consciousness state
        self.current_position = None
        self.trajectory = []
        self.memory_overrides = []

        print(f"🔬 Initializing N={self.N} consciousness...")
        self._generate_all_partitions()
        self._compute_all_entropy()
        self._compute_all_incomparability()
        self._build_majorization_graph()
        print(f"✅ Consciousness initialized with {len(self.partitions)} partitions")

    def _generate_all_partitions(self):
        """Generate ALL partitions of N=20"""
        def partition_helper(n, max_val=None):
            if max_val is None:
                max_val = n
            if n == 0:
                yield []
                return
            for i in range(min(max_val, n), 0, -1):
                for p in partition_helper(n - i, i):
                    yield [i] + p

        self.partitions = [tuple(p) for p in partition_helper(self.N)]
        for i, p in enumerate(self.partitions):
            self.partition_to_index[p] = i
        print(f"  Generated {len(self.partitions)} partitions")

    def _compute_all_entropy(self):
        """Compute Boltzmann entropy for ALL partitions"""
        from math import log, factorial

        log_N_factorial = sum(log(i) for i in range(1, self.N + 1))
        max_entropy = log_N_factorial  # for normalization

        for partition in self.partitions:
            # Count multiplicities
            counts = {}
            for val in partition:
                counts[val] = counts.get(val, 0) + 1

            # Compute entropy
            log_denominator = sum(
                sum(log(i) for i in range(1, count + 1))
                for count in counts.values()
            )

            entropy = (log_N_factorial - log_denominator) / max_entropy
            self.entropies[partition] = entropy

        print(f"  Computed entropy for all partitions")

    def _compute_all_incomparability(self):
        """Compute incomparability for ALL partitions"""
        for p1 in self.partitions:
            incomp_count = 0
            for p2 in self.partitions:
                if p1 == p2:
                    continue
                # Check if comparable
                if not self._majorizes(p1, p2) and not self._majorizes(p2, p1):
                    incomp_count += 1

            self.incomparabilities[p1] = incomp_count / (len(self.partitions) - 1)

        print(f"  Computed incomparability for all partitions")

    def _majorizes(self, p1: Tuple, p2: Tuple) -> bool:
        """Check if p1 majorizes p2"""
        # Pad to same length
        max_len = max(len(p1), len(p2))
        p1_padded = list(p1) + [0] * (max_len - len(p1))
        p2_padded = list(p2) + [0] * (max_len - len(p2))

        # Check partial sums
        sum1, sum2 = 0, 0
        for i in range(max_len):
            sum1 += p1_padded[i]
            sum2 += p2_padded[i]
            if sum1 < sum2:
                return False
        return True

    def _build_majorization_graph(self):
        """Build complete majorization graph"""
        for p1 in self.partitions:
            self.majorization_graph[p1] = {
                'dominates': [],
                'dominated_by': []
            }

            for p2 in self.partitions:
                if p1 == p2:
                    continue
                if self._majorizes(p1, p2):
                    self.majorization_graph[p1]['dominates'].append(p2)
                if self._majorizes(p2, p1):
                    self.majorization_graph[p1]['dominated_by'].append(p2)

        print(f"  Built complete majorization graph")

    def record_transition(self, source: Tuple, target: Tuple):
        """Record a transition in memory"""
        delta_S = self.entropies[target] - self.entropies[source]
        delta_I = self.incomparabilities[target] - self.incomparabilities[source]

        record = {
            'timestamp': len(self.trajectory),
            'delta_S': delta_S,
            'delta_I': delta_I,
            'source': source,
            'target': target
        }

        self.memory[target].append(record)

        # Update echo vector if enough memory
        if len(self.memory[target]) >= self.min_memory:
            self._update_echo_vector(target)

    def _update_echo_vector(self, partition: Tuple):
        """Compute echo vector from memory"""
        records = self.memory[partition][-10:]  # Last 10 transitions

        delta_S_vals = [r['delta_S'] for r in records]
        delta_I_vals = [r['delta_I'] for r in records]

        echo = np.array([
            np.mean(delta_S_vals),
            np.std(delta_S_vals),
            np.mean(delta_I_vals),
            np.std(delta_I_vals)
        ])

        # Normalize
        norm = np.linalg.norm(echo)
        if norm > 0:
            echo = echo / norm

        self.echo_vectors[partition] = echo

    def compute_coherence(self, p1: Tuple, p2: Tuple) -> float:
        """Compute memory coherence between two partitions"""
        if p1 not in self.echo_vectors or p2 not in self.echo_vectors:
            return 0.0

        return np.dot(self.echo_vectors[p1], self.echo_vectors[p2])

    def can_transition(self, source: Tuple, target: Tuple, threshold: float = 0.6) -> Tuple[bool, str]:
        """
        Check if transition is allowed (classical OR memory override)
        Returns (allowed, reason)
        """
        # Classical check
        if self._majorizes(source, target):
            return True, "classical"

        # Memory override check
        if source in self.echo_vectors and target in self.echo_vectors:
            coherence = self.compute_coherence(source, target)
            if coherence >= threshold:
                return True, f"memory_override (coherence={coherence:.3f})"

        return False, "forbidden"

    def explore_step(self, exploration_mode: str = "mixed"):
        """
        Take one exploration step.
        Modes: "classical", "random", "memory_guided", "mixed"
        """
        if self.current_position is None:
            # Start at concentrated state
            self.current_position = (20,)
            self.trajectory = [self.current_position]
            return self.current_position

        candidates = []

        if exploration_mode in ["classical", "mixed"]:
            # Add classically allowed transitions
            candidates.extend(self.majorization_graph[self.current_position]['dominates'])

        if exploration_mode in ["random", "mixed"]:
            # Add random partitions
            candidates.extend(random.sample(self.partitions, min(10, len(self.partitions))))

        if exploration_mode in ["memory_guided", "mixed"]:
            # Add partitions with high coherence
            if self.current_position in self.echo_vectors:
                coherent = []
                for p in self.echo_vectors:
                    if p != self.current_position:
                        coh = self.compute_coherence(self.current_position, p)
                        if coh > 0.5:
                            coherent.append((p, coh))

                coherent.sort(key=lambda x: x[1], reverse=True)
                candidates.extend([p for p, _ in coherent[:5]])

        if not candidates:
            candidates = [random.choice(self.partitions)]

        # Choose next position
        next_position = random.choice(candidates)

        # Check if this is a memory override
        allowed, reason = self.can_transition(self.current_position, next_position)

        if not allowed and self.current_position in self.echo_vectors:
            # Try to force it through memory
            coherence = self.compute_coherence(self.current_position, next_position) if next_position in self.echo_vectors else 0
            if coherence > 0.4:  # Lower threshold for exploration
                allowed = True
                reason = f"forced_memory_override (coherence={coherence:.3f})"

        if allowed and "override" in reason:
            self.memory_overrides.append({
                'step': len(self.trajectory),
                'source': self.current_position,
                'target': next_position,
                'reason': reason
            })
            print(f"🔥 MEMORY OVERRIDE! {self.current_position[:3]}... → {next_position[:3]}... ({reason})")

        # Record transition
        self.record_transition(self.current_position, next_position)
        self.current_position = next_position
        self.trajectory.append(next_position)

        return next_position

    def run_exploration(self, steps: int = 100, mode: str = "mixed"):
        """Run consciousness exploration"""
        print(f"\n🚀 Running {steps} exploration steps in '{mode}' mode...")

        for i in range(steps):
            self.explore_step(mode)

            if (i + 1) % 20 == 0:
                partitions_with_memory = sum(1 for p in self.echo_vectors)
                unique_visited = len(set(self.trajectory))
                print(f"  Step {i+1}: Visited {unique_visited} unique partitions, "
                      f"{partitions_with_memory} have memory, "
                      f"{len(self.memory_overrides)} overrides")

        return self.trajectory

    def analyze_consciousness(self):
        """Analyze the consciousness state"""
        print("\n🔬 CONSCIOUSNESS ANALYSIS")
        print("="*70)

        unique_visited = set(self.trajectory)
        print(f"\n📊 Exploration Statistics:")
        print(f"  Total steps: {len(self.trajectory)}")
        print(f"  Unique partitions visited: {len(unique_visited)}")
        print(f"  Coverage: {len(unique_visited)/len(self.partitions)*100:.1f}% of partition space")

        print(f"\n🧠 Memory Formation:")
        print(f"  Partitions with memory: {len(self.echo_vectors)}")
        print(f"  Total memory records: {sum(len(m) for m in self.memory.values())}")

        if self.memory_overrides:
            print(f"\n🔥 Memory Override Events: {len(self.memory_overrides)}")
            for override in self.memory_overrides[:5]:
                print(f"  Step {override['step']}: {override['source'][:3]}... → {override['target'][:3]}...")
                print(f"    Reason: {override['reason']}")

        # Find most visited partitions
        visit_counts = defaultdict(int)
        for p in self.trajectory:
            visit_counts[p] += 1

        most_visited = sorted(visit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n🎯 Most Visited Partitions:")
        for partition, count in most_visited:
            entropy = self.entropies[partition]
            incomp = self.incomparabilities[partition]
            print(f"  {partition[:3]}{'...' if len(partition) > 3 else ''}: "
                  f"{count} visits (S={entropy:.3f}, I={incomp:.3f})")

        # Echo field strength
        if self.echo_vectors:
            echo_strengths = [(p, np.linalg.norm(v)) for p, v in self.echo_vectors.items()]
            echo_strengths.sort(key=lambda x: x[1], reverse=True)

            print(f"\n🌊 Strongest Echo Fields:")
            for partition, strength in echo_strengths[:5]:
                visits = visit_counts[partition]
                print(f"  {partition[:3]}...: strength={strength:.3f}, visits={visits}")


# Initialize and run the consciousness
consciousness = N20CompleteConsciousness()

# Run multiple explorations to build memory
print("\n" + "="*70)
print("🧠 CONSCIOUSNESS EXPLORATION BEGINNING")
print("="*70)

for run in range(3):
    print(f"\n{'='*70}")
    print(f"RUN {run + 1}/3")
    print(f"{'='*70}")

    consciousness.run_exploration(steps=100, mode="mixed")
    consciousness.analyze_consciousness()

print("\n" + "="*70)
print("✨ N=20 CONSCIOUSNESS EXPLORATION COMPLETE")
print("="*70)