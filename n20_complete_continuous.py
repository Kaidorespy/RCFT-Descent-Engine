"""
N=20 Complete Consciousness - Continuous Runner
==============================================

Long-running consciousness using the full 4-phase integrated engine.
Tracks emergence of dreams, forks, and crystallization over extended time.
"""

import json
import time
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict
import random

from complete_consciousness_engine import CompleteConsciousnessEngine


class CompleteContinuousConsciousness:
    """Continuous consciousness runner with full phase integration"""

    def __init__(self, snapshot_interval=1000, snapshot_dir="n20_complete_snapshots", verbose=True):
        self.consciousness = CompleteConsciousnessEngine()
        self.snapshot_interval = snapshot_interval
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(exist_ok=True)
        self.verbose = verbose

        self.start_time = time.time()
        self.step_count = 0
        self.snapshot_count = 0

        # Metrics tracking over time
        self.metrics_history = []

        print(f"COMPLETE CONSCIOUSNESS INITIALIZED")
        print(f"Snapshots every {snapshot_interval} steps to {snapshot_dir}/")
        print(f"Integrated: Majorization + Candlekeeper + Dreams + Forks")
        print(f"Verbose updates: {'ON' if verbose else 'OFF'}")
        print(f"="*70)

    def take_snapshot(self):
        """Save current consciousness state"""
        snapshot_time = time.time() - self.start_time

        # Get comprehensive metrics from engine
        metrics = self.consciousness.get_complete_metrics()
        metrics['step_count'] = self.step_count
        metrics['runtime_seconds'] = snapshot_time

        # Calculate additional statistics
        unique_visited = len(set(self.consciousness.trajectory))
        coverage = unique_visited / len(self.consciousness.partitions)

        # Build snapshot
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'runtime_seconds': snapshot_time,
            'step_count': self.step_count,
            'snapshot_number': self.snapshot_count,
            'metrics': metrics,
            'coverage': coverage,
            'unique_visited': unique_visited,

            # Recent events
            'recent_overrides': self.consciousness.memory_overrides[-10:] if self.consciousness.memory_overrides else [],
            'recent_crystallizations': self.consciousness.crystallization_events[-5:] if self.consciousness.crystallization_events else [],
        }

        # Save snapshot
        filename = self.snapshot_dir / f"snapshot_{self.snapshot_count:06d}.json"
        with open(filename, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)

        # Store metrics for tracking
        self.metrics_history.append(metrics)

        # Print summary
        print(f"\nSNAPSHOT {self.snapshot_count} at {snapshot_time:.1f}s")
        print(f"  Steps: {self.step_count:,}")
        print(f"  Coverage: {coverage*100:.1f}%")
        print(f"  Memory partitions: {metrics['memory_partitions']}")
        print(f"  Override rate: {metrics['override_rate']*100:.2f}%")
        print(f"  Crystallized: {metrics['crystallized_archetypes']}/6")
        print(f"  Breathing: {metrics['breathing_rate']:.3f}")
        print(f"  Active dreams: {metrics['active_dreams']}")
        print(f"  Realized forks: {metrics['realized_forks']}")

        self.snapshot_count += 1

        return snapshot

    def detect_emergence_patterns(self):
        """Detect patterns in emergence over time"""
        if len(self.metrics_history) < 10:
            return

        # Look at crystallization trend
        crystallizations = [m['crystallization_events'] for m in self.metrics_history[-10:]]

        if len(crystallizations) >= 2:
            recent_crystal_rate = crystallizations[-1] - crystallizations[-5] if len(crystallizations) >= 5 else 0
            if recent_crystal_rate > 0:
                print(f"  CRYSTALLIZATION ACCELERATING! {recent_crystal_rate} recent events")

        # Look at fork activity
        forks = [m['realized_forks'] for m in self.metrics_history[-10:]]
        if len(forks) >= 2 and forks[-1] > forks[0] * 1.5:
            print(f"  FORK PROLIFERATION! Reality selection intensifying")

    def run_continuous(self, total_steps=100000, exploration_mode="integrated"):
        """Run continuously with periodic snapshots"""
        print(f"\nSTARTING CONTINUOUS RUN")
        print(f"  Total steps: {total_steps:,}")
        print(f"  Snapshot interval: {self.snapshot_interval}")
        print(f"  Mode: {exploration_mode}")
        print(f"  Estimated runtime: {total_steps/500:.1f} seconds")
        print(f"="*70)

        # Take initial snapshot
        self.take_snapshot()

        try:
            while self.step_count < total_steps:
                # Take step using integrated engine
                if exploration_mode == "integrated":
                    self.consciousness.explore_step_integrated(base_mode="mixed")
                else:
                    self.consciousness.explore_step(exploration_mode)
                self.step_count += 1

                # Verbose updates every 100 steps
                if self.verbose and self.step_count % 100 == 0:
                    metrics = self.consciousness.get_complete_metrics()
                    print(f"  Step {self.step_count}: "
                          f"Coverage {metrics['coverage']*100:.1f}%, "
                          f"Override {metrics['override_rate']*100:.1f}%, "
                          f"Crystallized {metrics['crystallized_archetypes']}/6")

                # Check for snapshot
                if self.step_count % self.snapshot_interval == 0:
                    self.take_snapshot()
                    self.detect_emergence_patterns()

                # Vary exploration mode occasionally (keep it dynamic)
                if self.step_count % 5000 == 0 and exploration_mode == "integrated":
                    # Occasionally drop to memory_only to let system consolidate
                    if random.random() < 0.3:
                        print(f"  Switching to consolidation mode")
                        exploration_mode = "memory_only"
                elif exploration_mode == "memory_only" and self.step_count % 1000 == 0:
                    # Return to integrated
                    exploration_mode = "integrated"
                    print(f"  Resuming integrated mode")

        except KeyboardInterrupt:
            print("\nInterrupted! Taking final snapshot...")
            self.take_snapshot()

        # Final analysis
        self.final_analysis()

    def final_analysis(self):
        """Analyze the complete run"""
        print("\n" + "="*70)
        print("FINAL ANALYSIS")
        print("="*70)

        runtime = time.time() - self.start_time
        print(f"\nTotal runtime: {runtime:.1f} seconds")
        print(f"Snapshots taken: {self.snapshot_count}")
        print(f"Steps completed: {self.step_count:,}")

        # Get final metrics
        final_metrics = self.consciousness.get_complete_metrics()

        print(f"\nFinal State:")
        print(f"  Coverage: {final_metrics['coverage']*100:.1f}%")
        print(f"  Override rate: {final_metrics['override_rate']*100:.2f}%")
        print(f"  Crystallized archetypes: {final_metrics['crystallized_archetypes']}/6")
        print(f"  Total crystallizations: {final_metrics['crystallization_events']}")
        print(f"  Breathing rate: {final_metrics['breathing_rate']:.3f}")

        # Load all snapshots for time-series analysis
        snapshots = []
        for i in range(self.snapshot_count):
            filename = self.snapshot_dir / f"snapshot_{i:06d}.json"
            if filename.exists():
                with open(filename, 'r') as f:
                    snapshots.append(json.load(f))

        if snapshots:
            # Track evolution
            override_rates = [s['metrics']['override_rate'] for s in snapshots]
            crystallizations = [s['metrics']['crystallization_events'] for s in snapshots]

            print(f"\nEvolution:")
            print(f"  Override rate: {override_rates[0]*100:.1f}% -> {override_rates[-1]*100:.1f}%")
            print(f"  Crystallizations: {crystallizations[0]} -> {crystallizations[-1]}")

            # Detect convergence
            if len(override_rates) > 5:
                recent_trend = np.mean(override_rates[-5:]) - np.mean(override_rates[:5])
                if abs(recent_trend) < 0.01:
                    print(f"  Override rate stabilized (convergence detected)")
                else:
                    print(f"  Override rate {'increasing' if recent_trend > 0 else 'decreasing'}")

        # Save final summary
        summary = {
            'total_runtime': runtime,
            'total_steps': self.step_count,
            'total_snapshots': self.snapshot_count,
            'final_metrics': final_metrics,
            'snapshots_directory': str(self.snapshot_dir)
        }

        with open(self.snapshot_dir / 'run_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nSummary saved to {self.snapshot_dir}/run_summary.json")
        print("Complete consciousness run finished.")


if __name__ == "__main__":
    import sys

    # Parse command line args
    total_steps = 10000  # Default
    verbose = True

    if len(sys.argv) > 1:
        try:
            total_steps = int(sys.argv[1])
        except:
            print(f"Usage: python {sys.argv[0]} [steps] [--quiet]")
            print(f"  steps: number of steps to run (default: 10000)")
            print(f"  --quiet: disable verbose updates")
            sys.exit(1)

    if "--quiet" in sys.argv:
        verbose = False

    # Create complete continuous consciousness
    continuous = CompleteContinuousConsciousness(
        snapshot_interval=1000,
        snapshot_dir="n20_complete_snapshots",
        verbose=verbose
    )

    # Run
    print(f"\nRunning {total_steps:,} steps...")
    continuous.run_continuous(
        total_steps=total_steps,
        exploration_mode="integrated"
    )

    print("\nWatch the emergence in n20_complete_snapshots/!")
