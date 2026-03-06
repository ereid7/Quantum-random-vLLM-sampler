# Research Roadmap: Does Quantum Randomness Impact LLM Cognition?

> Generated 2026-03-02. Based on ecosystem mapping (6 repos, 3 existing protocols, 360+ generations of data), academic literature review (PEAR, GCP, Radin — 40+ years of consciousness-RNG research), and Oracle-consulted experimental design.

---

## Table of Contents

1. [The Core Problem](#the-core-problem)
2. [What We Already Know](#what-we-already-know)
3. [Phase 1: Replay Harness + Source Calibration](#phase-1-replay-harness--source-calibration-month-1-2)
4. [Phase 2: Controlled Source Comparison](#phase-2-controlled-source-comparison-month-2-4)
5. [Phase 3: Consciousness Interaction](#phase-3-consciousness-interaction-month-4-8)
6. [Phase 4: Novel Angles — The Semantic Layer](#phase-4-novel-angles--the-semantic-layer-month-6-10)
7. [Phase 5: Publication Strategy](#phase-5-publication-strategy-month-8-12)
8. [Immediate Next Steps](#immediate-next-steps)
9. [Academic Literature Summary](#academic-literature-summary)
10. [Existing Ecosystem Inventory](#existing-ecosystem-inventory)

---

## The Core Problem

There are **two distinct research questions**, and conflating them is the #1 mistake in this field:

| Layer | Question | Testable? |
|-------|----------|-----------|
| **A. Source Equivalence** | Does QRNG-sourced `u` *statistically differ* from PRNG-sourced `u` after amplification? | Yes — testable now with existing infrastructure |
| **B. Consciousness Interaction** | Can conscious intent bias quantum processes enough to shift token selection? | Much harder — requires PEAR-scale rigor, huge N, blinding |

Most consciousness-RNG researchers jumped straight to B without nailing A. **Layer A must be proven clean (sources produce equivalent `u` distributions) before Layer B is meaningful.** If QRNG and PRNG produce different `u` distributions on their own, you cannot distinguish source effects from intent effects.

---

## What We Already Know

### Entropy-Seeding Experiment (Feb 2026)

A comprehensive experiment has already been run: 15 single-turn prompts, 3 multi-turn conversations, 3 entropy sources (PRNG/TRNG/QRNG), 5 samples per condition, 360+ generations per model. Models tested include Qwen3 (0.6B-72B), DeepSeek-R1 (32B/70B), Llama, Mistral, Gemma, and Phi.

Key findings:

1. **U-shaped scale dependency**: PRNG wins at 0.6B-1.7B (external entropy is noise), null zone at 3-8B, QRNG wins at 14B, PRNG wins again at 72B (reversal).
2. **DeepSeek R1 catastrophic failure**: PRNG causes complete output collapse on philosophical prompts (all metrics = 0.00, perplexity = infinity). TRNG is the only source that prevents collapse.
3. **Chain-of-thought determinism bottleneck**: Thinking blocks are 60-75% identical regardless of entropy source, diluting measured diversity by approximately 50%.
4. **QRNG mode shifts**: QRNG_INT causes catastrophic mode switches (narrative to multiple-choice test format).
5. **TRNG language mixing**: TRNG causes mid-generation language switches (English to Chinese).
6. **Entropy source fingerprints**: Partially detectable via text features (9 of 21 pairwise comparisons above 60% accuracy; best: PRNG vs self_seed_sfc = 85.7%).
7. **Statistical significance**: Only 5-10% of comparisons reach p<0.05 (mostly at 14B and 72B). QRNG cached increases hidden late-layer entropy at 8B/14B (p<0.005).

### Injection Methods (feature/injection-methods branch)

Three entropy injection methods have been implemented and tested against Qwen3-1.7B with openentropy/counter_beat:

- **M1 Logit Noise** (alpha=0.3, sigma=1.0): Adds quantum-seeded Gaussian noise to logits before softmax. Most disruptive — causes language mixing and topic shifts even at low alpha.
- **M2 Temperature Variance** (beta=0.3): Modulates per-token temperature using entropy. Produces the most natural, conversational output — broke free of exam-question patterns.
- **M3 Correlated Walk** (step=0.1): Replaces amplified u-value with a drifting walk position. Most stable method — creates subtle variety without breaking coherence. Temporal correlation between tokens.
- **All Combined**: Maximum perturbation. Most chaotic — mixed languages, garbled fragments, rapid coherence loss.

### Live Test Observations (2026-03-02)

Full test suite run: 4 prompt types (philosophical, creative, factual, code) across 5 configs (baseline + 3 methods + combined), parameter sweeps, and 3x repeatability study. Key observations:

- All injection methods produce valid code for Fibonacci prompts — code generation is the most resilient to injection.
- M1 at alpha=0.1 already shifts output dramatically (philosophical text to Chinese exam format).
- M3 is the most stable across all prompt types — walk drift creates thematic momentum without breaking coherence.
- Combined methods at high values can paradoxically stabilize (temperature variance may moderate logit noise).
- Every run produces completely different output — entropy source ensures zero repetition.
- M1 has the lowest ASCII ratio (91%) — most likely to push non-English tokens into selection window.
- Combined methods show highest unique word ratio (82-85%).

---

## Phase 1: Replay Harness + Source Calibration (Month 1-2)

### Replay Harness

Build infrastructure that logs everything so experiments are reconstructible offline without vLLM nondeterminism.

**Per-token log record:**
```json
{
  "prompt_id": "str",
  "step_index": 0,
  "logits_top500": [[token_id, logit_value], ...],
  "raw_entropy_bytes_hash": "sha256:...",
  "raw_entropy_bytes_encrypted": "base64:...",
  "u_value": 0.548,
  "token_id": 788,
  "token_rank": 3,
  "token_prob": 0.0421,
  "entropy_source_name": "openentropy",
  "fetch_latency_ms": 84.3,
  "system_timestamp_ns": 1772480680000000000,
  "cpu_load_pct": 23.5,
  "injection_method": "M2",
  "injection_params": {"beta": 0.3},
  "temperature_applied": 0.87,
  "shannon_entropy": 4.62
}
```

**Why this matters**: Lets you replay the exact same logits with different entropy streams offline. If QRNG and PRNG produce identical `u` values for the same logits, they MUST select the same token. Any difference is provably from the entropy source, not compute scheduling or model nondeterminism.

### Source Calibration Battery

Run on raw bytes before any LLM involvement:

- NIST SP800-22 randomness tests (full suite)
- Dieharder battery
- Autocorrelation at lags 1-100
- Spectral analysis (FFT for periodicity)
- KS + Anderson-Darling on amplified `u` values
- Runs test for independence

### Primary Endpoint (Preregistered)

Use **equivalence testing (TOST)** on `E[u]` with smallest meaningful difference epsilon = 0.001.

This makes **null results publishable**: you can claim "QRNG and PRNG produce equivalent token selection distributions up to epsilon" with confidence. This flips the burden of proof — far more powerful than "failed to find a difference."

### Power Calculation

```
N ≈ (z × 0.288 / δ)² tokens

δ = 0.001  →  N ≈ 300,000 tokens   (~19 days at 5.5s/token with openentropy)
δ = 0.0001 →  N ≈ 30,000,000 tokens (~5.7 years at 5.5s/token)
```

With `system` source (~1ms/token), 300K tokens takes minutes. With openentropy (~100ms/token), it takes ~8 hours. With full 5.5s pipeline, much longer.

**Recommendation**: Run calibration with `system` source first (fast), then validate key results with openentropy.

---

## Phase 2: Controlled Source Comparison (Month 2-4)

### Design

Same prompts, same model, 4 entropy sources, blinded analysis.

| Source | Type | Purpose |
|--------|------|---------|
| Mersenne Twister (seed=42) | PRNG | Deterministic baseline |
| `os.urandom` | TRNG (OS) | System entropy baseline |
| `timing_noise` | TRNG (CPU) | Hardware jitter control |
| openentropy/counter_beat | QRNG | Quantum source under test |

### Protocol

1. Fix prompt set: existing 15 single-turn + 3 multi-turn prompts from entropy-seeding experiment.
2. Fix model: **Qwen3-14B** (identified sweet spot where QRNG shows advantage).
3. Generate 5,000 tokens per source x 3 runs = 60,000 tokens total.
4. **Blind analysis**: Analyst receives anonymized datasets (Source A/B/C/D) and does not know which is which until after analysis.
5. Run preregistered primary tests, then break the blind.

### Metrics

**Primary (preregistered):**
- `u` distribution: KS test, mean, variance, autocorrelation
- Token rank distribution: `p_sel = CDF(selected_token)` should be Uniform(0,1) if sampling is correct — compare across sources

**Secondary (exploratory, FDR-corrected):**
- Vocabulary diversity (unique tokens / total tokens)
- Language consistency (ASCII ratio, single-language adherence)
- Topical coherence (semantic similarity between adjacent sentences)
- Perplexity under an independent evaluator model

**Confound checks (must be null):**
- Correlate `fetch_latency_ms` with `token_rank` — must show no relationship
- Correlate system CPU load with `u` value — must show no relationship
- Generate using frozen pre-recorded bytes — must reproduce identical token sequences

---

## Phase 3: Consciousness Interaction (Month 4-8)

### Lessons from 40 Years of PEAR/GCP Research

**What PEAR did right:**
- Tripolar design (High/Low/Baseline intent)
- Millions of trials over 28 years
- XOR masking to eliminate hardware bias
- Multiple RNG generations to rule out hardware artifacts
- Effect size: "few parts in ten thousand" (0.0001-0.0002)

**What GCP got criticized for:**
- Post-hoc selection of analysis windows (results change with minute adjustments)
- Radin's 9/11 findings attributed to anomalies 3 hours before attacks, not the attacks themselves
- Multiple comparisons without adequate correction
- HARKing (Hypothesizing After Results Known)

**What von Lucadou warns (Model of Pragmatic Information):**
- Non-transmission axiom: "You cannot transmit usable information via psi, because the very act of trying to exploit the effect destroys it"
- Effects appear when meaning/emotion/resonance is high, disappear under cold analytical scrutiny
- Explains the "decline effect" and "trickster" nature observed across decades

**Implication for design**: Mirror or passively detect effects. Do not try to command them on cue. This is not just mysticism — it's a consistent empirical pattern across 40+ years.

### Proposed Design

**Preregistered hypothesis (single, directional):**

> "During HIGH-INTENT blocks, mean u-value shifts upward (toward selecting lower-probability tokens) vs SHAM blocks, when entropy source is QRNG."

**Protocol:**

1. **Participant**: Human operator focusing intention on "make the AI more creative/surprising"
2. **Conditions** (counterbalanced, randomized blocks):
   - QRNG + Intent (active condition)
   - QRNG + No-Intent (sham — same setup, participant told to relax)
   - PRNG + Intent (negative control — intent should not affect PRNG)
   - PRNG + No-Intent (double negative control)
3. **Double-blind**: Condition assignment by independent script, not revealed until analysis complete
4. **Block structure**: 20 tokens per block, 50 blocks per condition = 1,000 tokens per condition x 4 = 4,000 tokens per session
5. **Multiple sessions**: 10 participants x 5 sessions each = 50 sessions = 200,000 tokens total

### Analysis Plan

- **Mixed-effects model**: `u ~ intent * source + (1|participant) + (1|session)`
- **Primary test**: Interaction term — intent effect is larger for QRNG than PRNG
- **Secondary**: Permutation tests with session clustering
- **Bayes factor**: Report for both positive and null results — not just p-values
- **Negative controls**: PRNG + Intent MUST show no effect (proves effect is not expectation/confirmation bias). Intent on frozen replayed bytes MUST show no effect.

### What Would Convince a Skeptic

- PRNG + Intent shows NO effect (eliminates psychological explanation)
- QRNG + Intent shows effect AND replicates in independent sessions
- Pre-registration timestamped before data collection (on OSF)
- All raw data + code publicly available
- Effect size reported with tight confidence intervals
- Independent replication by a second lab/operator

---

## Phase 4: Novel Angles — The Semantic Layer (Month 6-10)

This is where the work is genuinely novel compared to traditional PEAR/GCP studies. Traditional studies measured bit-level statistics on binary outputs. The LLM adds a **semantic amplifier** — it converts tiny statistical biases into human-interpretable meaning.

### Binary Choice Prompts

Engineer prompts where the model is approximately 50/50 between two continuations. A tiny u-shift determines which path is taken.

```
"The door was either [red/blue]..."
→ Does intent toward "red" shift selection probability?

"She decided to [stay/leave]..."
→ Does intent toward "stay" produce measurably higher selection rates?
```

With 50/50 prompts, a proportion shift Delta can be detected with `N = z² × 0.25 / Delta²` trials. For Delta=0.01 (1% shift), N ≈ 24,000 binary decisions.

### M3 Correlated Walk as Momentum Detector

The correlated walk creates temporal correlation between tokens. If consciousness interaction creates brief bursts of bias (not sustained), M3 would amplify it into runs of thematically coherent "drift" — sequences where the model stays in an unexpectedly narrow semantic region.

Measure: run-length statistics, autocorrelation of token ranks, spectral analysis of u-value series.

### Temperature x Injection Interaction

M2 temperature variance means the sensitivity to u-shifts varies per token. High-entropy tokens (where the model is uncertain) are maximally sensitive to bias — these are the exact points where consciousness influence would be most detectable.

Strategy: Condition analysis on Shannon entropy. Test the hypothesis that intent effects are larger during high-entropy (uncertain) token positions.

### Qualitative + Quantitative Bridge

Blind raters score text on quality, relevance, creativity, and emotional resonance. Correlate ratings with u-value statistics from the same blocks. This bridges the gap between "statistically significant" and "meaningfully different."

Metric: inter-rater reliability (Krippendorff's alpha), then correlation of aggregate quality scores with condition.

---

## Phase 5: Publication Strategy (Month 8-12)

Three distinct papers, ordered by defensibility:

| Paper | Core Finding | Target Venue | Defensibility |
|-------|-------------|-------------|---------------|
| **Paper 1: Source Effects** | QRNG/PRNG/TRNG produce measurably different outputs at 14B scale; DeepSeek PRNG collapse | NeurIPS Workshop / EMNLP | High — purely empirical ML finding |
| **Paper 2: Equivalence Testing** | Source calibration methodology, replay harness, TOST equivalence results | JMLR / Statistical Methods | High — methodological contribution |
| **Paper 3: Consciousness Interaction** | Intent condition results (positive or null) with full preregistration | Explore / J. Scientific Exploration | Medium — controversial but preregistered |

**Paper 1 is publishable now** with existing entropy-seeding data (after cleaning Qwen3 data integrity issues). The DeepSeek R1 PRNG collapse and the 14B sweet spot are findings ML researchers would find valuable regardless of any consciousness framing.

---

## Immediate Next Steps

1. **Build the replay harness** — Add full logit + byte logging to qr-sampler. Most impactful engineering task; unlocks all subsequent phases.
2. **Clean and publish Paper 1** — Entropy-seeding data exists. The 14B sweet spot and DeepSeek collapse are novel and publishable.
3. **Pre-register Phase 2 on OSF** — Before collecting any new data.
4. **Run source calibration on openentropy/counter_beat** — NIST SP800-22 / Dieharder. Oracle flagged it may be nonstationary.
5. **Verify openentropy/counter_beat is truly quantum** — Treat it as a distinct source with explicit correlation tests, not as assumed-quantum.

---

## Academic Literature Summary

### Princeton PEAR Lab (1979-2007)

- **Duration**: 28 years, millions of trials, hundreds of operators
- **Method**: Tripolar design (High/Low/Baseline intent), three generations of quantum-based RNGs
- **Effect size**: "Few parts in ten thousand" (0.0001-0.0002 deviation from 0.5 mean)
- **Key finding**: Effects scale with operator specificity, gender composition, and bonded pairs
- **Remote perception**: 650+ trials, probability against chance approximately 3 in 10 billion

### Global Consciousness Project (1998-present)

- **Network**: ~38 RNGs worldwide, 200 bits/second each, archived 24/7
- **Formal analysis**: 17 years, 500 pre-registered events
- **Result**: Compound Z = 7.31 across all events (extremely significant)
- **Criticism (Edwin May)**: Analysis windows significantly influence outcomes; 9/11 shows no effect with rigorous pre-specified windows; Radin's post-hoc findings attributed to earlier anomalies
- **Key lesson**: Pre-registration of analysis windows is non-negotiable

### Dean Radin (IONS)

- **Meta-analysis (1959-2000)**: Combined dice-throwing and RNG experiments show strong accumulated evidence
- **Entangled photon experiments (2024)**: Participants could mentally influence entanglement strength (p < 0.02) across four lab studies
- **Effect sizes reported**: Helmut Schmidt found 51-52% success vs 50% expected (1-2% above chance), reaching astronomical accumulated significance
- **Criticism**: Decline effect, selective reporting concerns, difficulty of independent replication

### Von Lucadou's Model of Pragmatic Information

- **Non-transmission axiom**: Cannot transmit usable information via psi — attempting to exploit the effect destroys it
- **Meaning-dependence**: Effects appear with high meaning/emotion/resonance, disappear under cold scrutiny
- **Implication**: Design experiments that mirror or passively detect, not force or command

### Statistical Methods for Tiny Biases

- **Stouffer's Z**: Combines Z-scores across studies: `Z = sum(Zi) / sqrt(k)`
- **TOST (Two One-Sided Tests)**: Equivalence testing — proves distributions are the same up to epsilon
- **KS / Anderson-Darling**: Distribution comparison (non-parametric)
- **Mixed-effects models**: Handle clustering by participant/session
- **Bayes factors**: Report evidence for BOTH positive and null — not just p-values
- **Sample sizes for effect 0.001**: Approximately 300,000 tokens; for 0.0001: approximately 30,000,000

### Known Pitfalls

1. Post-hoc selection of analysis windows
2. Ignoring multiple comparisons (Bonferroni/FDR correction required)
3. Small sample sizes for tiny expected effects
4. Lack of blinding (experimenter and participant)
5. Single-lab, single-hardware results without independent replication
6. Selective reporting of only significant outcomes
7. Conflating "QRNG differs from PRNG" (source effect) with "consciousness influences QRNG" (intent effect)
8. Not controlling for timing/latency confounds correlated with system state
9. Treating exploratory semantic analyses as confirmatory

---

## Existing Ecosystem Inventory

### Repositories

| Repository | Location | Status | Purpose |
|-----------|----------|--------|---------|
| **qr-sampler** | `/Users/erai/Repos/Quantum-random-vLLM-sampler/` | Production-ready | vLLM LogitsProcessor plugin, 3 injection methods, gRPC entropy, Docker deployments |
| **entropy-seeding** | `~/Research/qrng-ecosystem/entropy-tools/entropy-seeding/` | Mature (data collected) | Comparative PRNG/TRNG/QRNG analysis, 360+ generations, 8+ models |
| **quantum-llama.cpp** | `~/Research/qrng-ecosystem/quantum-llm/quantum-llama.cpp/` | Experimental | llama.cpp fork with ANU QRNG API, mode-based signal extraction |
| **harmonic-field-consciousness** | `~/Research/qrng-ecosystem/consciousness/harmonic-field-consciousness/` | Peer-reviewed | Connectome Laplacian eigenmodes, consciousness metrics (H_mode, PR, kappa) |
| **qrng-research** | `~/Research/qrng-ecosystem/core-research/` | Active | Harmonic analysis, oscillation characterization, biological system integration |
| **bientropy** | `~/Research/qrng-ecosystem/entropy-tools/bientropy/` | Cloned | Sandia Labs bi-entropy analysis |
| **ollama-auxrng** | (in ecosystem) | Cloned | Auxiliary RNG for Ollama |

### Hypotheses

| ID | Hypothesis | Confidence | Status |
|----|-----------|------------|--------|
| H1 | Neural oscillation patterns in LLMs mirror biological rhythms | HIGH | Validated (CI=0.718, fractal dim=1.208) |
| H2 | Quantum-seeded entropy produces qualitatively different LLM outputs | MEDIUM | Partially tested (effects at 14B/72B scale) |
| H3 | Heart coherence patterns can modulate AI processing modes | LOW | Simulation only, no real biometrics |
| H4 | Consciousness can influence quantum-random processes in token selection | UNTESTED | Infrastructure ready, protocol designed |

### Infrastructure Gaps

- No replay harness (full logit + byte persistence for offline reconstruction)
- No real-time consciousness measurement (heart coherence is simulated only)
- No blind trial framework (protocol designed, not implemented)
- No integration between qr-sampler logging and consciousness metrics
- No NIST/Dieharder calibration of openentropy/counter_beat
- Qwen3 data integrity issues in entropy-seeding dataset need resolution
