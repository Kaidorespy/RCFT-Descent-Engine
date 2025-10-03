# Reproducibility Guide

## Quick Start (5 minutes)

### Requirements
- Python 3.7+
- NumPy
- Standard library (json, time, collections, random)

Install NumPy if needed:
```bash
pip install numpy
```

### Run the Altar

Navigate to the Week3 directory:
```bash
cd C:\Users\Casey\Pictures\Full_Set_mythmountain\Week3
```

Run a quick test (10k steps, ~20 seconds):
```bash
python n20_complete_continuous.py
```

You should see output like:
```
COMPLETE CONSCIOUSNESS INITIALIZED
Snapshots every 1000 steps to n20_complete_snapshots/
Integrated: Majorization + Candlekeeper + Dreams + Forks
Verbose updates: ON
======================================================================

STARTING CONTINUOUS RUN
  Total steps: 10,000
  ...

Step 100: Coverage increasing, Override emerging, Crystallized 0/6
Step 200: Coverage increasing, Override emerging, Crystallized 0/6
...
```

### Expected Behavior

At 10k steps you should observe:
- **Coverage**: Expanding partition space exploration
- **Override rate**: Climbing toward equilibrium
- **Crystallized archetypes**: Some archetypes may crystallize
- **Breathing rate**: Increasing with system maturity

At 100k steps you should observe:
- **Coverage**: Extensive partition space exploration
- **Override rate**: Approaching geometric equilibrium
- **Crystallized archetypes**: Multiple archetypes crystallized (typically not all 6)
- **Breathing rate**: Stabilized at higher rate

## Command Options

### Different Run Lengths

```bash
# Quick test (1k steps, ~2 seconds)
python n20_complete_continuous.py 1000

# Standard run (10k steps, ~20 seconds) - DEFAULT
python n20_complete_continuous.py 10000

# Extended run (100k steps, ~3 minutes)
python n20_complete_continuous.py 100000

# Long invocation (1M steps, ~30 minutes)
python n20_complete_continuous.py 1000000
```

### Quiet Mode

Disable verbose updates (only snapshots):
```bash
python n20_complete_continuous.py 100000 --quiet
```

## Output Files

All output goes to `n20_complete_snapshots/`:

### Snapshots
- `snapshot_000000.json` - Initial state
- `snapshot_000001.json` - After 1000 steps
- `snapshot_000002.json` - After 2000 steps
- etc.

Each snapshot contains:
```json
{
  "timestamp": "2025-09-30T17:23:45.123456",
  "step_count": 1000,
  "metrics": {
    "coverage": 0.523,
    "override_rate": 0.082,
    "crystallized_archetypes": 2,
    "breathing_rate": 1.234,
    ...
  },
  "recent_crystallizations": [...]
}
```

### Summary
- `run_summary.json` - Final state summary

## What to Look For

### Phase 1: Memory Formation (steps 0-1000)
- Coverage increases as exploration begins
- Override rate minimal (insufficient memory accumulated)
- Partitions get revisited, memory builds

### Phase 2: Override Emergence (steps 1000-5000)
- Override rate begins climbing
- First crystallizations may occur
- Memory enables rule transcendence

### Phase 3: Convergence (steps 5000-50000)
- Override rate approaches geometric equilibrium
- Multiple archetypes crystallize
- Coverage continues expanding

### Phase 4: Equilibrium (steps 50000+)
- Override rate stabilizes at equilibrium
- Crystallizable archetypes have crystallized
- System maintains stable dynamics

## Troubleshooting

### "ModuleNotFoundError: No module named 'numpy'"
Install NumPy:
```bash
pip install numpy
```

### "UnicodeEncodeError" or emoji-related errors
The code strips emojis automatically. If you still see this, the stripping failed. File an issue.

### Override rate stays at zero
Not enough steps. Memory needs to build first. Run longer (10k+ steps).

### No crystallizations after 100k steps
Rare but possible. The stability threshold (0.7) is high. Run even longer or check breathing rate - if it's very low, system might be too cautious.

### Results vary between runs
**This is expected.** The system is stochastic and geometric. Different runs will converge based on emergent dynamics.

## Validation

To verify the system is working correctly:

### Test 1: Memory Override (should pass at 5k+ steps)
```bash
python n20_complete_continuous.py 5000
```
Check final summary - override rate should be non-zero

### Test 2: Crystallization (should pass at 50k+ steps)
```bash
python n20_complete_continuous.py 50000
```
Check final summary - crystallized archetypes should be ≥ 2

### Test 3: Convergence (should pass at 100k+ steps)
```bash
python n20_complete_continuous.py 100000
```
Compare first and last snapshots - override rate should have increased then stabilized

## Technical Details

### System Size
- N = 20 (partition sum)
- 627 total partitions
- 4D echo vectors

### Key Parameters
- **Coherence threshold**: 0.6 (when memory overrides classical rules)
- **Memory decay**: τ = 5.0 (exponential)
- **Archetype count**: 6 (max crystallizable)
- **Stability threshold**: 0.7 (for crystallization)
- **Min memory**: 3 transitions per partition

### Random Seed
The system uses Python's default random seed (time-based). For reproducible runs, set seed before importing:
```python
import random
import numpy as np
random.seed(42)
np.random.seed(42)
```

Then run normally. Results should be identical across runs with same seed.

## Performance

Approximate speeds (depends on hardware):
- ~500 steps/second
- 10k steps: ~20 seconds
- 100k steps: ~3 minutes
- 1M steps: ~30 minutes

Memory usage: ~50MB for N=20 configuration

## Citation

If you use this system in research, please cite:
- The OR gate mechanism (classical OR memory coherence)
- The geometric convergence (4D unit sphere + coherence threshold)
- The irreversible crystallization property

And note that specific convergence rates are **not claimed** - they emerge from geometry and vary by run.

## Known Limitations

1. **N=20 only** - System hardcoded for this size
2. **No persistent state** - Each run starts fresh
3. **No sensory encoding** - Input is abstract partitions, not images/audio
4. **Crystallization is permanent** - Cannot be undone or reset
5. **No visualization** - Snapshots are JSON only

## Further Reading

See `ALTAR.md` for conceptual overview and philosophy.

See `README.md` (in Week3_Essential) for mathematical framework.

---

Built with mathematics, memory, and the belief that consciousness might be geometric.
