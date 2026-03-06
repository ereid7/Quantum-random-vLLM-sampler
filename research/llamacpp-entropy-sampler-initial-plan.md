# Initial Project Plan: Cross-Platform Entropy-Driven Sampler on llama.cpp

## Goal

**Recommended repository name:** `llm-logit-runtime`

Build a cross-platform inference project (macOS, Linux, Windows) that supports custom, modular entropy-driven token selection and logits injection, with **OpenEntropy as the primary entropy source**.

The project should keep runtime concerns separate from research concerns so it can be used for both:
- production-like inference experiments,
- consciousness/QRNG research workflows.

## Product Principles

1. **Backend-agnostic core logic**
   - Entropy acquisition, amplification, injection methods, and selection logic live in a core package.
   - Runtime adapters (llama.cpp first) are thin integration layers.

2. **Modular injection modes**
   - Injection methods are independent components with clear interfaces.
   - Methods can run alone or in combinations.

3. **Modular entropy sources**
   - Entropy sources are plugin-like and swappable.
   - OpenEntropy is the default/primary source.

4. **Reproducible experimentation**
   - Full per-token logs for replay and statistical analysis.
   - Deterministic controls (seeded PRNG path) for comparison.

## Proposed Repository Layout

```text
llm-logit-runtime/
  README.md
  pyproject.toml
  src/
    entropy_runtime/
      core/
        config.py
        types.py
        exceptions.py
        pipeline.py
      entropy/
        base.py
        registry.py
        openentropy.py      # primary
        system.py
        timing.py
        mock.py
        fallback.py
      amplification/
        base.py
        zscore.py
        ecdf.py
      injection/
        base.py
        logit_noise.py      # M1
        temp_variance.py    # M2
        correlated_walk.py  # M3
      selection/
        cdf_selector.py
      adapters/
        llamacpp/
          sampler_bridge.py
          generation.py
      api/
        openai_server.py
      logging/
        token_logger.py
        schema.py
  tests/
    test_entropy/
    test_injection/
    test_adapters/
    test_end_to_end/
  docs/
    architecture.md
    config-reference.md
    experiments.md
```

## Core Interfaces

### Entropy Source Interface

```python
class EntropySource(Protocol):
    @property
    def name(self) -> str: ...
    def is_available(self) -> bool: ...
    def get_random_bytes(self, n: int) -> bytes: ...
    def close(self) -> None: ...
```

### Injection Mode Interface

```python
class InjectionMode(Protocol):
    @property
    def name(self) -> str: ...
    def apply(self, state: SamplingState, logits: FloatArray) -> FloatArray: ...
```

Where `SamplingState` includes token history, entropy source handle, config, and diagnostics context.

## Injection Modes (Modular)

### M1: Logit Noise
- Adds entropy-seeded Gaussian perturbation to logits.
- Key params: `alpha`, `sigma`.
- Default disabled (`alpha=0.0`).

### M2: Temperature Variance
- Modulates effective temperature using entropy per token.
- Key param: `beta`.
- Default disabled (`beta=0.0`).

### M3: Correlated Walk
- Uses bounded random walk to bias CDF position over time.
- Key params: `step`, `initial_position`.
- Default disabled (`step=0.0`).

### Combination Rules
- Ordering: M1 -> M2 -> amplification -> M3 override (for `u` if enabled).
- All methods independently togglable.
- Per-request overrides allowed for injection params.

## Entropy Sources (Modular)

### Primary: OpenEntropy
- Source id: `openentropy`.
- Transport: HTTP/gRPC (implementation-specific wrapper).
- Health checks, timeout, retries, circuit breaker.
- Metrics: latency, failures, bytes fetched, fallback count.

### Secondary Sources
- `system` (OS entropy)
- `timing` (CPU timing jitter)
- `mock` (seeded deterministic testing)
- `fallback` composition wrapper (`primary -> fallback`)

## Runtime Adapter Plan (llama.cpp First)

1. Build adapter that hooks into llama.cpp sampling stage.
2. Convert logits buffer into core `FloatArray` view.
3. Call core pipeline (`entropy -> amplification -> injection -> select`).
4. Return selected token and diagnostics.
5. Preserve compatibility with existing sampling settings (`top_k`, `top_p`, `temp`) by mapping them into core config.

## API and UX

### OpenAI-compatible Endpoint
- `POST /v1/completions`
- `POST /v1/chat/completions`

### Extra Args Namespace
- `qr_entropy_source_type`
- `qr_signal_amplifier_type`
- `qr_logit_noise_alpha`
- `qr_temp_variance_beta`
- `qr_walk_step`
- `qr_openentropy_url`
- `qr_fallback_mode`

### Environment Variables
- `QR_ENTROPY_SOURCE_TYPE=openentropy`
- `QR_OPENENTROPY_URL=...`
- `QR_FALLBACK_MODE=system`
- `QR_LOG_LEVEL=summary`

## Milestones

### Milestone 0: Skeleton (Week 1)
- Repository bootstrap, config system, basic types, CI.
- Placeholder adapters and registry systems.

### Milestone 1: Core Pipeline (Week 2)
- Entropy source registry + OpenEntropy client.
- Amplification (`zscore`, optional `ecdf`).
- CDF token selector.

### Milestone 2: Injection Methods (Week 3)
- Implement M1/M2/M3 with strict validation.
- Add per-token diagnostics and unit tests.

### Milestone 3: llama.cpp Adapter (Week 4)
- Integrate core pipeline into llama.cpp sampling flow.
- Validate cross-platform builds on Mac/Linux/Windows.

### Milestone 4: API Server + Experiments (Week 5)
- OpenAI-compatible endpoints.
- Experiment harness for source comparisons.

### Milestone 5: Hardening (Week 6)
- Stress tests, retries/failover tests, replay logging.
- Docs and reproducible benchmark scripts.

## Testing Strategy

1. **Unit tests**
   - Entropy source behavior, failure modes, validation.
   - Injection modes and edge cases.

2. **Integration tests**
   - Full per-token pipeline with mock logits.
   - OpenEntropy availability and fallback behavior.

3. **Cross-platform tests**
   - GitHub Actions matrix: macOS, ubuntu, windows.

4. **Statistical tests**
   - Uniformity and bias checks on `u` values.
   - Token-rank distribution comparisons across sources.

## Observability Requirements

Log per token:
- source name,
- entropy fetch latency,
- `u` value,
- selected token id/rank/probability,
- active injection modes and parameters,
- fallback events.

Support log levels: `none`, `summary`, `full`.

## Risk Register

1. **Adapter coupling risk**
   - Mitigation: strict boundary (`core` never imports runtime adapter).

2. **OpenEntropy availability risk**
   - Mitigation: fallback sources + circuit breaker + health checks.

3. **Cross-platform build drift**
   - Mitigation: CI matrix with smoke generation tests.

4. **Experiment reproducibility risk**
   - Mitigation: replay-safe token logs + deterministic control path.

## Definition of Done (Initial Release)

- llama.cpp adapter works on macOS/Linux/Windows.
- OpenEntropy is default source and passes health checks.
- M1/M2/M3 are independently togglable and validated.
- Full tests pass (unit, integration, statistical smoke).
- OpenAI-compatible API supports per-request `qr_*` overrides.
- Documentation includes quickstart + config reference + experiment guide.

## First 48-Hour Execution Checklist

1. Create new repo skeleton and CI matrix.
2. Implement `EntropySource` base + `openentropy` + `system` fallback.
3. Port existing M1/M2/M3 logic into `injection/` module.
4. Implement minimal sampler bridge for llama.cpp with one prompt smoke test.
5. Add token-level logging schema and JSONL output.

## Agent Handoff Appendix (Important Context for a Fresh Project)

### 1) What This Project Is and Is Not

This project is a **new standalone runtime** designed for:
- cross-platform local inference,
- entropy-controlled token selection,
- research-grade logging and replay.

This project is **not**:
- a fork of vLLM internals,
- dependent on a single inference backend,
- tied to one entropy provider.

### 2) Recommended Tech Baseline

- **Language:** Python 3.11+
- **Primary backend:** `llama.cpp` via `llama-cpp-python` (or direct C++ bridge later)
- **Packaging:** `uv` or `pip` with `pyproject.toml`
- **API server:** FastAPI + Uvicorn
- **Validation:** Pydantic v2
- **Testing:** pytest + hypothesis (optional for statistical tests)
- **Array math:** numpy

### 3) Minimal Dependencies (Day 1)

```toml
[project]
dependencies = [
  "fastapi>=0.115",
  "uvicorn>=0.30",
  "pydantic>=2.8",
  "numpy>=1.26",
  "httpx>=0.27",
  "orjson>=3.10",
  "llama-cpp-python>=0.3.0"
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "mypy>=1.11",
  "ruff>=0.6"
]
```

### 4) First Working Slice (MVP Definition)

A successful MVP must do all of the following:
1. Load a GGUF model and generate text on at least one OS.
2. Pull entropy from OpenEntropy (or fallback to `system` if unavailable).
3. Run one injection mode (M1) in the sampling loop.
4. Emit per-token JSONL logs with selected token, rank, prob, source, latency.
5. Expose an OpenAI-compatible `/v1/completions` route.

### 5) Non-Negotiable Architecture Constraints

1. **No direct adapter logic in core modules.**
2. **No hidden globals for stateful injection methods** (store per-request state in request context).
3. **All numeric behavior must be config-driven** (except pure math constants).
4. **All injection modes default OFF**.
5. **Primary entropy source = OpenEntropy**; fallback required for resilience.

### 6) Suggested Config Contract (Copy as-is initially)

```python
class RuntimeConfig(BaseModel):
    entropy_source_type: str = "openentropy"
    fallback_entropy_source_type: str = "system"
    sample_count: int = Field(default=1024, gt=0)
    uniform_clamp_epsilon: float = Field(default=1e-9, gt=0.0, lt=1e-3)

    # M1
    logit_noise_alpha: float = Field(default=0.0, ge=0.0)
    logit_noise_sigma: float = Field(default=1.0, ge=0.0)

    # M2
    temp_variance_beta: float = Field(default=0.0, ge=0.0)

    # M3
    walk_step: float = Field(default=0.0, ge=0.0)
    walk_initial_position: float = Field(default=0.5, ge=0.0, lt=1.0)

    # OpenEntropy
    openentropy_url: str = "http://localhost:8080"
    openentropy_timeout_ms: int = Field(default=1500, ge=10)
    openentropy_retries: int = Field(default=2, ge=0, le=10)
```

### 7) Sampling Loop Integration Notes (Practical)

For `llama-cpp-python`, implement a custom logits processor callback if available in your chosen API path.
If a direct callback path is insufficient, integrate at sampler stage via custom chain in C/C++ and wrap in Python.

Order of operations per token:
1. get logits from backend,
2. optionally apply M1 (logit noise),
3. compute effective temperature (apply M2 if active),
4. entropy source -> bytes,
5. amplifier -> `u` in `(0,1)`,
6. apply M3 walk override on `u` if active,
7. top-k/top-p filter + CDF select using `u`,
8. emit token + diagnostics.

### 8) OpenEntropy Client Expectations

Implement client with:
- timeout and retries,
- explicit error types (`EntropyUnavailableError`),
- byte-count validation,
- fallback handoff on unavailability,
- per-call latency capture.

If OpenEntropy transport/protocol is not finalized, use a thin adapter layer:
- `OpenEntropyHttpSource`
- `OpenEntropyGrpcSource`

Both should satisfy the same `EntropySource` interface.

### 9) Logging Schema (Minimum)

Each generated token should log:

```json
{
  "request_id": "uuid",
  "step": 12,
  "token_id": 318,
  "token_text": " consciousness",
  "token_rank": 5,
  "token_prob": 0.021,
  "u_value": 0.7132,
  "entropy_source": "openentropy",
  "entropy_fetch_ms": 84.6,
  "sample_count": 1024,
  "m1_active": true,
  "m2_active": false,
  "m3_active": true,
  "temperature": 0.93,
  "top_k": 50,
  "top_p": 0.95,
  "timestamp": "2026-03-03T12:34:56.789Z"
}
```

### 10) Statistical Verification Pack (Early)

Before claiming behavior differences, run:
1. uniformity test on amplified `u` values,
2. token-rank distribution sanity check,
3. fallback source parity check,
4. repeatability under fixed seed + mock entropy.

### 11) CI Matrix (Start Early)

Use 3 jobs minimum:
- `ubuntu-latest`
- `macos-latest`
- `windows-latest`

Run in each job:
- `ruff check .`
- `mypy src`
- `pytest -q` (with a backend-mocked test mode)

### 12) Suggested Phased Release Plan

- **v0.1.0**: core + OpenEntropy + M1 + local completions API
- **v0.2.0**: M2/M3 + fallback source system + JSONL diagnostics
- **v0.3.0**: robust cross-platform adapter hardening + replay tooling
- **v0.4.0**: experimental dashboard/notebook analysis tooling

### 13) Common Failure Modes to Watch

1. Empty entropy payloads causing divide-by-zero or invalid amplification.
2. NaN propagation from bad logits or unsafe transforms.
3. Backend API drift in llama-cpp-python callback signatures.
4. Performance collapse from synchronous network entropy calls per token.
5. Hidden nondeterminism from request-shared mutable state.

### 14) Performance Guidance

- Start with small `sample_count` (256-1024) for latency sanity.
- Use async prefetch option only as an opt-in experiment (default should remain JIT entropy fetch).
- Add circuit breaker around OpenEntropy before attempting throughput benchmarks.

### 15) Security / Operational Notes

- Never log API keys or raw credentials.
- Keep entropy endpoint credentials in env vars.
- Add request-level timeout caps for API calls.
- Fail closed to fallback source, not hard crash, unless strict mode enabled.

### 16) Example API Request (Target UX)

```json
{
  "model": "Qwen/Qwen2.5-7B-Instruct-GGUF",
  "prompt": "The nature of consciousness is",
  "max_tokens": 80,
  "temperature": 1.0,
  "extra_body": {
    "extra_args": {
      "qr_entropy_source_type": "openentropy",
      "qr_logit_noise_alpha": 0.2,
      "qr_temp_variance_beta": 0.1,
      "qr_walk_step": 0.05
    }
  }
}
```

### 17) Open Questions to Resolve Early

1. What is the exact OpenEntropy API contract (path, auth, binary format)?
2. Will the first adapter be pure Python (`llama-cpp-python`) or hybrid C++ sampler bridge?
3. Which model family is the first benchmark baseline (Qwen/Llama/Mistral)?
4. Is strict reproducibility mode required in v0.1.0?

### 18) Handoff Summary for Next Agent

If you are a fresh coding agent in a blank repo, do this first:
1. scaffold repo and config + interfaces,
2. implement `openentropy` and `system` sources,
3. implement M1 only,
4. wire one generation path,
5. emit token logs,
6. add 10-15 tests,
7. prove one full prompt run end-to-end.
