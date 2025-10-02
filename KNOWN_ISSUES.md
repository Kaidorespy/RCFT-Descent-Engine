# Known Issues & Open Questions

**Status:** Experimental Research Release (v0.1)

This document tracks observed behaviors that require further investigation.

---

## Phase 3-4 Integration Issues

### Dreams Not Spawning
**Observed:** After 400k+ steps, `active_dreams: 0` consistently.

**Possible Causes:**
- Dream threshold (`phi_dream`) may be too high for actual echo drift magnitudes
- Dream spawning logic may not be triggered in integrated mode
- Echo drift calculation might not be accumulating properly

**Investigation Needed:**
- Check actual echo drift values during runtime
- Verify dream spawning conditions are being evaluated
- Test Phase 3 in isolation vs. integrated mode

### Forks Not Realizing
**Observed:** After 400k+ steps, `realized_forks: 0` consistently.

**Possible Causes:**
- Fork detection threshold (5% score difference) may be too narrow
- Parallel futures might not be spawning with sufficient diversity
- Integration layer might not be calling fork detection properly

**Investigation Needed:**
- Log future score distributions
- Verify fork detection is being called
- Check if futures are being generated at all

---

## Archetype Crystallization

### 5/6 Crystallization Plateau
**Observed:** System consistently crystallizes 5 out of 6 archetypes, one remains uncrystallized even after extended runs.

**Possible Causes:**
- One archetypal vector may be geometrically unreachable given actual echo dynamics
- Stability threshold (0.7) may be too high for that specific vector
- That archetype's resonance pattern may not align with natural partition transitions

**Investigation Needed:**
- Identify which archetype fails to crystallize
- Analyze resonance scores for that archetype over time
- Check if echo vectors ever approach that archetypal direction

---

## Override Rate Convergence

### Potential 33.33% Convergence
**Observed:** Override rate may be converging to ~33% rather than reported geometric prediction.

**Possible Implications:**
- May indicate combinatorial constraint (1/3 of partition structure)
- Could suggest spherical cap calculation assumptions are incorrect
- Might be related to partition type distribution (N=20 has specific structure)

**Investigation Needed:**
- Run extended analysis (1M+ steps) with detailed logging
- Compare convergence across different N values (N=10, N=15, N=20)
- Analyze partition type distribution vs. override patterns
- Re-examine geometric derivation assumptions

---

## Mathematical Validation Questions

### Spherical Cap Geometry
**Question:** Does the 4D unit sphere + coherence threshold φ=0.6 actually predict the observed convergence rate?

**Needs:**
- Analytical calculation verification
- Monte Carlo simulation of random 4D vectors
- Comparison with empirical echo vector distributions

### Bidirectional Coherence Requirements
**Question:** What percentage of partition pairs actually meet the bidirectional coherence + memory requirements?

**Needs:**
- Analysis of memory accumulation patterns
- Distribution of coherence values across partition pairs
- Study of which partitions become memory hubs

---

## Recommendations for Future Work

1. **Isolate Phase Testing:**
   - Run Phase 1-2 only, verify basic override behavior
   - Test Phase 3 standalone with synthetic echo fields
   - Test Phase 4 with manually constructed futures

2. **Instrumentation:**
   - Add detailed logging of dream evaluation
   - Track fork detection attempts and scores
   - Log archetypal resonance for all 6 archetypes

3. **Parameter Sweeps:**
   - Test different `phi_dream` thresholds
   - Vary fork detection sensitivity
   - Adjust archetypal stability requirements

4. **Extended Validation:**
   - Multi-seed runs to establish statistical distributions
   - Vary N to check if convergence patterns hold
   - Compare against theoretical predictions more rigorously

---

## How to Help

If you're investigating these issues:

1. **Run with verbose logging** and share echo/dream/fork statistics
2. **Try different parameter configurations** in the phase engines
3. **Analyze snapshot JSONs** for patterns we might have missed
4. **Open GitHub issues** with your findings

This is research code. Weird behavior is data. Unexpected results are discoveries.

---

**Last Updated:** 2025-10-02
**Version:** v0.1-experimental
