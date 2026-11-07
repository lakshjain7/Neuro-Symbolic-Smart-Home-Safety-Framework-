# Research & Prototype Plan

Status: draft v1. Hardware is being built; nothing here assumes a specific meter model.
One topology question is still open (see section 2.1), so the design is parameterized to cover both readings.

---

## 1. Goal and hypotheses

Build a neuro-symbolic safety layer for smart-home power telemetry that (a) detects real hazards and (b) does not raise alarms caused by sensor faults. We measure **both** error directions: true positives (real hazards caught) and false positives (sensor glitches that got through).

| ID | Hypothesis | Evidence needed |
|---|---|---|
| H1 | The physics guardrail reduces sensor-fault false alarms without a meaningful drop in recall on real hazards | Confusion matrix: NN alone vs NN+guardrail vs guardrail alone |
| H2 | Using graph structure beats the same model without it | Ablation: no graph / fully connected / wiring graph / learned graph |
| H3 | Learned relations match the true wiring and the planted relations | Edge precision/recall vs ground truth, null scenario control |
| H4 | The system can say *which* sensor is faulty | Localization accuracy (voting / residual attribution) |
| H5 | Cold start and device swaps are handled without false alarms | Time-to-stable; false alarm rate after a swap |
| H6 | The guardrail's detection floor is a measurable function of load size and meter noise | Detection-floor curve; replication on AMPds at scale |

## 2. Testbed

### 2.1 Open decision (blocks the generator registry, not the generator code)

"Parallel meters at each socket" has two readings:

- **A. Siblings:** 3 separate outlets per zone, each metered. 9 independent device meters in 3 zones. No parent meter, so energy conservation is not checkable unless one meter per zone is moved to a series (parent) position or a board-level meter is added. Strength: parallel branches share supply voltage, so several meters measure the same voltage and a disagreeing meter can be identified by majority vote (supports H4).
- **B. Redundant:** 3 meters on the same load per zone. Only 3 independent loads, but sensor faults on real hardware can be labelled by disagreement between redundant meters (ground truth with no injection).

The registry (section 4.1) encodes topology as data, so either reading is a config change.

### 2.2 Facts to check when the hardware is ready

- Do the three sockets share one circuit/breaker? This decides whether cross-zone coupling can serve as a weak-edge control.
- Which fields does each meter report (V, I, P, Q/S, pf)? The identity rules need them.
- Meter accuracy specs, sampling rate, clock source.
- Rated limit of every extension box and socket.

## 3. End-to-end pipeline

```
measure -> annotate -> preprocess -> model (neural) -> guardrail (symbolic) -> evaluate
              ^                                                                   |
              +------------------ generator / simulator (ground truth) -----------+
```

### 3.1 Measurement protocol

- One clock for all nodes; timestamps checked at the gateway.
- Sampling rate set by the fastest event of interest (inrush, spikes), not by convenience.
- Raw data is immutable; metadata stored beside it (device, zone, rating, firmware, calibration).
- Sessions: noise-floor session, long normal sessions, scripted event sessions.

### 3.2 Annotation schema

Two separate axes: true cause and sensor state.

```
start, end, node, cause, sensor_state, source, severity, session_id
cause         : normal | overload | stuck_on | abnormal_cycle | ...
sensor_state  : healthy | spike | stuck_at | dropout | drift | field_corrupt | disconnect
source        : physical | injected
```

Rules:
- Events are logged live (CLI/button), not reconstructed from memory.
- Software faults are injected on a clean copy; the clean copy stays as ground truth.
- Scoring is event-level with latency. Point-adjusted scoring is not used.

### 3.3 Preprocessing

- Resample to a fixed rate; explicit gap policy (forward-fill vs mask, recorded in metadata).
- Normalize with training-set statistics only; power also expressed relative to node rating (scale-invariant features).
- Chronological, session-blocked splits. Never shuffle windows across the split.
- Window length chosen from event timescales.
- Hold out whole fault types to test generalization.

## 4. Generator / loader

Purpose: drive the testbed (or a simulator of it) with realistic, seeded, reproducible load scenarios; impose known relations; inject anomalies; and write ground-truth labels. It is also the main dry-run tool before hardware exists.

### 4.1 Device registry (manifest)

```yaml
topology:
  zones:
    - id: zone_a
      circuit: board_1          # fill after wiring check
      meters:
        - id: m_a1
          appliance: heater     # declared type (optional, may be null)
          rated_w: 1500
          controllable: relay   # relay | manual
          parent: null          # set for hierarchy variants
          profile: {levels_w: [0, 1500], pf_class: resistive}
        - id: m_a2
          appliance: fan
          rated_w: 60
          controllable: manual
  redundancy_groups: []         # reading B: [[m_x1, m_x2, m_x3]]
limits:
  max_simultaneous_w_per_zone: <fill from rating>
```

The registry carries two kinds of information that must stay separate:
- **Prior knowledge** (zones, wiring): may be given to the model as graph structure.
- **Evaluation labels** (planted relations): never visible to the model.

### 4.2 Relation templates

| Type | Meaning | Parameters |
|---|---|---|
| trigger | A on -> B on after delay | delay, probability, jitter |
| exclusion | A and B not on together | window |
| routine | A follows a time-of-day pattern | time distribution |
| containment | zone contains meters | from wiring |
| natural coupling | voltage sag, thermal effects | measured, not planted |

Planted relations are stochastic (probability and jitter) so a deterministic pattern cannot be memorized.

### 4.3 Components

1. **Scenario compiler:** registry + templates + routine distributions -> seeded schedule of on/off events. Routine timing can be sampled from real usage statistics (e.g. AMPds activations).
2. **Actuation:** relay control, or CLI prompt for manual devices. Each actuation is verified from the meter (planned on == measured on), and mismatches are flagged.
3. **Anomaly layer:** (a) safe physical anomalies, (b) software sensor faults on a clean copy, (c) hardware faults by design (disconnect a meter).
4. **Ground-truth logger:** device states, scenario id, seed, planted relations, anomalies.
5. **Safety envelope:** per-zone and per-socket load caps enforced when the schedule is compiled, before anything is switched.
6. **Software twin / simulator:** synthesizes power from device profiles with the same interface. Used for pipeline development and for the **null scenario** (no planted relations) that checks the model does not invent links.

### 4.4 Output layout

```
data/sessions/<session_id>/
  manifest.yaml     # registry snapshot
  schedule.json     # compiled events + seed
  raw.csv           # measurements (or simulator output)
  labels.csv        # annotation schema in 3.2
  relations.json    # planted relations (evaluation only)
```

### 4.5 Acceptance tests for the generator

- Same seed -> identical schedule.
- No schedule violates the safety envelope.
- Simulator output respects device profiles (power levels, pf class).
- Null scenario contains no planted relation entries.
- Relation recovery metrics computable end to end on simulator data.

## 5. Algorithm research tracks

### 5.1 Symbolic layer (guardrail)

Calibrated, not trained. Tolerance `tol = a + b*|P|`, with `a` and `b` from the noise-floor session.

| Rule | Needs | Notes |
|---|---|---|
| R1 electrical identity: `P ~ V*I*pf`, `S^2 ~ P^2 + Q^2` | V, I, P, pf/Q | per meter, no identity required |
| R2 shared-voltage consistency | meters on one supply | majority vote localizes the faulty meter |
| R3 rated-limit bounds | rated_w | |
| R4 conservation: parent ~ sum of children + residual | a parent meter | only if topology provides one |
| R5 mismatch discrimination | R4 or R2 | persistent offset vs spike; avoid blocking real leakage/ground faults |

Open research question: a hard veto can suppress a real hazard (a ground fault also breaks conservation). R5 must separate "sensor fault" from "real fault that breaks the sum".

### 5.2 Neural layer

Baseline ladder, each rung must be beaten before the next is justified:
threshold/z-score -> Isolation Forest -> PCA residual -> LSTM autoencoder -> per-node forecaster (no graph) -> graph model.

Graph variants: none / fully connected / wiring graph / learned adjacency / learned with physical prior. Temporal variants: GRU, TCN, Transformer.
Task: unsupervised on normal data (reconstruction/forecast residual); labels used only for threshold calibration and evaluation.
Multi-resolution option: zone layer over device layer, with a coherence constraint between levels.

### 5.3 Integration modes (compare at least two)

post-hoc veto; input gating; physics penalty in the loss; probabilistic fusion (fault vs real-event likelihood).

### 5.4 Identity and initialization

Three identities kept separate: node (hardware id), location (zone/circuit), appliance type (optional).
Flow: registration -> commissioning window (learn per-node baseline, device-agnostic rules active) -> operation -> change-point handling (re-enter learning on device swap).
Zone inference from electrical data (voltage correlation) tested against declared zones.

## 6. Experiments

| ID | Experiment | Hypothesis |
|---|---|---|
| E0 | Noise-floor session: all meters on one load | calibration |
| E1 | Long normal baseline | H5 |
| E2 | Injected sensor faults, full catalogue | H1, H4 |
| E3 | Safe real anomalies | H1 |
| E4 | Planted relation recovery + null scenario | H3 |
| E5 | Cold start: hours until stable | H5 |
| E6 | Device swap | H5 |
| E7 | Zone discovery vs declared zones | H3 |
| E8 | Detection floor vs load size | H6 |
| E9 | Replication on AMPds (21 devices) with injected faults | H6 |
| E10 | Reading B only: real sensor-fault ground truth by redundancy | H1, H4 |

Metrics: precision, recall, F1, latency, per fault type; results across sessions with confidence intervals; ablation table for every component.

## 7. Known issues in the current repo

1. Cell 37 of `Graph_Transformer.ipynb` downsamples AMPds (already 1-minute data) by 60 and labels it "1 minute"; windows are actually hours long.
2. The `PhysicsGuardrail` exists only as a README snippet; no notebook cell implements it.
3. README accuracy figures (92%, 94.3%, 88.7%, ...) are not produced by any cell: no labels or evaluation code exist.
4. README names A3TGCN2 and 6 anomaly classes; the notebook trains a GCN + Transformer reconstruction model.
5. `requirements.txt` and `LICENSE` are referenced but missing.
6. The notebook has a hard-coded OpenAI key placeholder cell; move to environment variables.

## 8. Order of work

1. Resolve the open decision in 2.1.
2. Implement generator, registry schema and simulator (section 4) with tests.
3. Build the preprocessing module and the evaluation harness; rehearse on simulator data and on AMPds with injected faults.
4. Implement the symbolic layer R1-R3 (R4/R5 conditional on topology) with tests.
5. Implement the baseline ladder, then the graph model.
6. Hardware: E0 -> E1 -> E2/E3 -> E4 onward.
7. Update README claims to match measured results only.
