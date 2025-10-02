# Known Issues & Open Questions

**Status:** Experimental Research Release (v0.1)

This document tracks observed behaviors that require further investigation.

---

## Phase 3-4 Integration Status

### Dreams - ✅ Working
**Status:** Dreams spawn and decay naturally

**Observed Behavior:**
- Dreams spawn during early exploration (typically 5-10 active dreams)
- Dreams decay over time with exponential falloff
- Active dream count drops to 0 as unconfirmed dreams fade

**This is expected behavior.** Dreams are temporary projections that fade unless repeatedly confirmed by real transitions.

### Fork Tracking Data Type Mismatch
**Status:** ⚠️ Working but metrics may be slightly inaccurate

**Observed:** Phase 4 `realized_history` contains strings instead of `RealizedFuture` objects.

**Impact:**
- Forks ARE working (realities are being selected)
- Fork count metrics may be inflated by ~10-20%
- Cannot distinguish "ghost" forks (unrealized) from actual realized forks

**Current Workaround:**
- Code checks if entries have `is_ghost` attribute
- If not (strings), counts them anyway
- System remains functional, just less precise tracking

**Root Cause:**
- Somewhere in Phase 4, fork history is storing fork IDs (strings) instead of full objects
- Need to trace `realized_history.append()` calls in `phase4_echo_forking.py`

**To Fix:**
- Find where `realized_history` is populated
- Ensure it stores `RealizedFuture` objects, not string IDs

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
