# Adaptive Trust Implementation Plan

This document outlines the tasks needed to modify the current implementation to support adaptive (dynamic) trust calculation.

## Overview

The current implementation uses **static trust scores** (computed once). This plan adds **adaptive trust calculation** that updates trust scores dynamically based on performance over multiple rounds.

## Implementation Phases

### Phase 1: Core Trust Manager (High Priority)

**Goal**: Create the central TrustManager component

**Tasks**:
1. ✅ Create `TrustManager` class in `federated_server.py`
   - Initialize with configuration parameters (alpha, beta, decay rates)
   - Store trust history for all clients
   - Implement `update_trust(client_id, performance_metrics)` method

2. ✅ Implement `TrustHistory` data structure
   - Store: trust_scores list, timestamps, performance_metrics
   - Track: round numbers, consistency metrics
   - Metadata: initial_trust, last_updated, update_count

3. ✅ Implement trust update formula
   - Weighted moving average: `trust_new = α × trust_old + (1-α) × new_performance`
   - Multi-factor aggregation (optional enhancement)
   - Configurable weights

4. ✅ Add trust bounds validation
   - Ensure trust stays in [0, 1] range
   - Apply smoothing to prevent oscillations
   - Handle edge cases (NaN, infinity)

**Files to Modify**:
- `src/federated_server.py` (add TrustManager class)

**New Files**:
- `src/trust_manager.py` (optional: separate file for TrustManager)

---

### Phase 2: Trust History Storage (High Priority)

**Goal**: Persist trust history across rounds

**Tasks**:
1. ✅ Create trust history storage system
   - Choose storage method (file-based JSON, SQLite, or in-memory)
   - Design storage structure for client trust data

2. ✅ Implement `save_trust_history()` method
   - Save trust history to disk
   - Include timestamps and metadata

3. ✅ Implement `load_trust_history()` method
   - Load existing trust history on startup
   - Handle missing/corrupted files gracefully

4. ✅ Add trust history directory structure
   - `results/trust_history/client_1/trust_scores.json`
   - `results/trust_history/client_1/performance_metrics.json`

**Files to Modify**:
- `src/federated_server.py` (TrustManager class)

**New Directories**:
- `results/trust_history/`

---

### Phase 3: Enhanced Client Tracking (High Priority)

**Goal**: Clients track and report performance history

**Tasks**:
1. ✅ Add `performance_history` attribute to `FederatedClient`
   - List of dictionaries: `[{round: 1, val_acc: 0.85, ...}, ...]`
   - Track validation accuracy per round

2. ✅ Modify `FederatedClient.compute_trust()` 
   - Keep current behavior for backward compatibility
   - Store performance metrics in history

3. ✅ Enhance `FederatedClient.get_model_update()`
   - Include performance history in update
   - Add consistency metrics (variance, trend)
   - Include current round number

4. ✅ Add consistency score calculation
   - Calculate variance of recent performance
   - Detect trends (improving, declining, stable)
   - Compute consistency metric

**Files to Modify**:
- `src/federated_client.py`

---

### Phase 4: Multi-Round Support (High Priority)

**Goal**: Enable multiple federated learning rounds

**Tasks**:
1. ✅ Modify `ExperimentRunner` to support rounds
   - Add `num_rounds` parameter (default: 1 for backward compatibility)
   - Create round loop structure

2. ✅ Update experiment flow
   - Phase 1: Local training (all clients)
   - Phase 2: Trust update (TrustManager updates scores)
   - Phase 3: Aggregation (using updated trust)
   - Phase 4: Distribution (send global model back)

3. ✅ Integrate TrustManager into experiment
   - Initialize TrustManager at start
   - Call trust update after each round
   - Pass updated trust to aggregator

4. ✅ Handle round-by-round state
   - Maintain client state across rounds
   - Track round numbers
   - Handle client dropouts/rejoins

**Files to Modify**:
- `experiment.py` (ExperimentRunner class)

---

### Phase 5: Trust Decay & Anomaly Detection (Medium Priority)

**Goal**: Add advanced trust management features

**Tasks**:
1. ✅ Implement trust decay mechanism
   - Time-based decay (trust decreases if not updated)
   - Participation-based decay (penalty for missing rounds)
   - Configurable decay rate

2. ✅ Add anomaly detection
   - Detect sudden trust drops (> threshold)
   - Identify suspicious patterns
   - Flag clients for investigation

3. ✅ Implement trust recovery
   - How quickly trust increases after improvement
   - Recovery rate configuration
   - Gradual vs. rapid recovery

**Files to Modify**:
- `src/federated_server.py` (TrustManager class)

---

### Phase 6: Initial Trust & Cold Start (Medium Priority)

**Goal**: Handle new clients joining the system

**Tasks**:
1. ✅ Implement initial trust assignment
   - Options: neutral (0.5), low (0.3), or first-round performance
   - Configurable initial trust strategy

2. ✅ Add cold start handling
   - New clients with no history
   - Bootstrap trust from first round
   - Gradual trust building

3. ✅ Handle client identification
   - Unique client IDs
   - Track which clients are new vs. existing

**Files to Modify**:
- `src/federated_server.py` (TrustManager class)
- `src/federated_client.py`

---

### Phase 7: Integration with Aggregator (Medium Priority)

**Goal**: Use updated trust scores in aggregation

**Tasks**:
1. ✅ Modify `TrustAwareAggregator.aggregate()`
   - Accept trust scores from TrustManager (not from client updates)
   - Use updated trust scores for weighting
   - Handle trust score updates between rounds

2. ✅ Update aggregation flow
   - Get trust scores from TrustManager
   - Apply trust weights to client contributions
   - Ensure trust scores are current

**Files to Modify**:
- `src/federated_server.py` (TrustAwareAggregator class)

---

### Phase 8: Configuration & Parameters (Low Priority)

**Goal**: Make trust calculation configurable

**Tasks**:
1. ✅ Create trust configuration structure
   - Alpha (history weight): default 0.7
   - Beta (new performance weight): default 0.3
   - Decay rate: default 0.95
   - Window size: default 5 rounds
   - Anomaly threshold: default 0.2

2. ✅ Add configuration file/parameters
   - `config/trust_config.json` or
   - Command-line arguments or
   - Class initialization parameters

3. ✅ Document configuration options
   - Explain each parameter
   - Provide recommended values
   - Show impact of different settings

**Files to Create**:
- `config/trust_config.json` (optional)

**Files to Modify**:
- `experiment.py` (add config loading)

---

### Phase 9: Visualization & Analysis (Low Priority)

**Goal**: Visualize trust evolution

**Tasks**:
1. ✅ Add trust evolution plots
   - Time series: trust score vs. round number
   - Multiple clients on same plot
   - Show trends and patterns

2. ✅ Create trust statistics visualization
   - Distribution of trust scores
   - Trust changes over time
   - Anomaly highlights

3. ✅ Update existing visualizations
   - Include trust history in reports
   - Show trust trends in summary

**Files to Modify**:
- `src/visualization.py`

---

### Phase 10: Logging & Monitoring (Low Priority)

**Goal**: Track trust changes for debugging and analysis

**Tasks**:
1. ✅ Add trust update logging
   - Log when trust changes
   - Log reason for change (performance, decay, etc.)
   - Include before/after values

2. ✅ Create trust change audit trail
   - Track all trust modifications
   - Store in results/trust_history/audit.log

3. ✅ Add trust statistics tracking
   - Mean, std, min, max trust per round
   - Store in results summary

**Files to Modify**:
- `src/federated_server.py` (TrustManager)
- `src/evaluation.py`

---

### Phase 11: Testing & Validation (High Priority)

**Goal**: Ensure adaptive trust works correctly

**Tasks**:
1. ✅ Unit tests for TrustManager
   - Test trust update formula
   - Test decay mechanism
   - Test bounds validation
   - Test anomaly detection

2. ✅ Integration tests
   - Multi-round federated learning
   - Trust evolution scenarios
   - Client performance changes

3. ✅ Backward compatibility tests
   - Single round (num_rounds=1) should work as before
   - Existing code should not break

4. ✅ Performance tests
   - Trust update speed
   - Storage efficiency
   - Memory usage

**Files to Create**:
- `tests/test_trust_manager.py`
- `tests/test_adaptive_trust_integration.py`

---

### Phase 12: Documentation (Medium Priority)

**Goal**: Document adaptive trust features

**Tasks**:
1. ✅ Update README.md
   - Explain adaptive trust concept
   - Configuration options
   - Usage examples

2. ✅ Add code documentation
   - Docstrings for TrustManager
   - Explain trust update formula
   - Document data structures

3. ✅ Create usage examples
   - Multi-round experiment
   - Trust configuration
   - Trust visualization

**Files to Modify**:
- `README.md`
- Code docstrings

---

## Implementation Order (Recommended)

### Sprint 1: Foundation (Phases 1-3)
- Core TrustManager
- Trust history storage
- Enhanced client tracking

### Sprint 2: Integration (Phases 4, 7)
- Multi-round support
- Aggregator integration

### Sprint 3: Advanced Features (Phases 5-6)
- Trust decay
- Anomaly detection
- Cold start handling

### Sprint 4: Polish (Phases 8-12)
- Configuration
- Visualization
- Testing
- Documentation

---

## Key Design Decisions

### 1. Backward Compatibility
- Default `num_rounds=1` maintains current behavior
- Existing single-round experiments continue to work

### 2. Storage Strategy
- Start with file-based JSON (simple)
- Can upgrade to database later if needed

### 3. Trust Update Formula
- Start with simple weighted moving average
- Can enhance with multi-factor later

### 4. Configuration
- Use class initialization parameters first
- Add config file later if needed

---

## Testing Strategy

### Unit Tests
- TrustManager.update_trust() with various scenarios
- Trust decay calculations
- Bounds validation
- Anomaly detection

### Integration Tests
- Complete multi-round experiment
- Trust evolution over 10 rounds
- Client performance improvement scenario
- Client performance degradation scenario

### Regression Tests
- Single round still works (backward compatibility)
- Existing visualizations still work
- Results format unchanged for single round

---

## Success Criteria

1. ✅ Trust scores update dynamically over multiple rounds
2. ✅ Trust history is persisted and can be loaded
3. ✅ Clients track performance history
4. ✅ Multi-round experiments work end-to-end
5. ✅ Backward compatibility maintained (single round)
6. ✅ Trust evolution can be visualized
7. ✅ Configuration is flexible and documented
8. ✅ All tests pass

---

## Notes

- Start simple: basic weighted moving average
- Add complexity gradually: multi-factor, decay, etc.
- Maintain backward compatibility throughout
- Test thoroughly at each phase
- Document as you go

---

## Estimated Effort

- Phase 1-3: 2-3 days (core functionality)
- Phase 4, 7: 1-2 days (integration)
- Phase 5-6: 1-2 days (advanced features)
- Phase 8-12: 2-3 days (polish)
- **Total: ~6-10 days** for complete implementation
