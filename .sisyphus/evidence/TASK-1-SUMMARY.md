# TASK 1: Add 6 New Config Fields for Injection Methods

## Objective
Add 6 new configuration fields to `QRSamplerConfig` in `src/qr_sampler/config.py` for injection methods (M1, M2, M3), and add all 6 field names to the `_PER_REQUEST_FIELDS` frozenset.

## Changes Made

### 1. Updated `_PER_REQUEST_FIELDS` frozenset (lines 23-47)
Added 6 new field names to the frozenset:
- `"logit_noise_alpha"`
- `"logit_noise_sigma"`
- `"temp_variance_beta"`
- `"walk_step"`
- `"walk_initial_position"`
- `"injection_verbose"`

**Total fields in frozenset: 21** (15 original + 6 new)

### 2. Added 6 New Fields to QRSamplerConfig (lines 199-224)

#### M1: Logit Noise Injection
```python
logit_noise_alpha: float = Field(
    default=0.0,
    description="M1: Gaussian logit noise magnitude. 0 = disabled.",
)
logit_noise_sigma: float = Field(
    default=1.0,
    description="M1: Standard deviation of Gaussian noise before scaling by alpha.",
)
```

#### M2: Temperature Variance Injection
```python
temp_variance_beta: float = Field(
    default=0.0,
    description="M2: Temperature modulation magnitude. 0 = disabled.",
)
```

#### M3: Correlated Walk Injection
```python
walk_step: float = Field(
    default=0.0,
    description="M3: Correlated walk step size. 0 = disabled.",
)
walk_initial_position: float = Field(
    default=0.5,
    description="M3: Initial walk position in [0, 1).",
)
```

#### Logging Control
```python
injection_verbose: bool = Field(
    default=False,
    description="Log injection method activity at each token.",
)
```

## Verification Results

✅ **Syntax Validation**: PASSED (py_compile)
✅ **Field Definitions**: All 6 fields correctly defined with proper defaults
✅ **_PER_REQUEST_FIELDS**: All 6 field names present in frozenset
✅ **Field Placement**: Correctly positioned between "Token Selection" and "Logging" sections
✅ **Pattern Compliance**: All fields follow existing Field() pattern with description
✅ **Type Hints**: Correct types (float, bool) with proper defaults

## Configuration Usage

### Environment Variables (auto-supported by pydantic-settings)
```bash
export QR_LOGIT_NOISE_ALPHA=0.1
export QR_LOGIT_NOISE_SIGMA=1.0
export QR_TEMP_VARIANCE_BETA=0.05
export QR_WALK_STEP=0.01
export QR_WALK_INITIAL_POSITION=0.5
export QR_INJECTION_VERBOSE=true
```

### Per-Request Overrides (via extra_args)
```python
extra_args = {
    "qr_logit_noise_alpha": 0.1,
    "qr_logit_noise_sigma": 1.0,
    "qr_temp_variance_beta": 0.05,
    "qr_walk_step": 0.01,
    "qr_walk_initial_position": 0.5,
    "qr_injection_verbose": True,
}
```

## Architecture Compliance

✅ **No hardcoded values**: All numeric constants are config fields
✅ **Per-request overridable**: All 6 fields in `_PER_REQUEST_FIELDS`
✅ **Immutable config**: Uses pydantic BaseSettings
✅ **Naming conventions**: 
  - Environment variables: `QR_*` prefix
  - Extra args keys: `qr_*` prefix
✅ **Documentation**: All fields have descriptive Field(description=...) text

## Files Modified
- `src/qr_sampler/config.py`
  - Lines 23-47: Updated `_PER_REQUEST_FIELDS` frozenset
  - Lines 199-224: Added "Injection methods" section with 6 new fields

## Status
✅ **TASK COMPLETE**

All expected outcomes met:
- [x] 6 new fields added to QRSamplerConfig
- [x] All 6 field names added to _PER_REQUEST_FIELDS frozenset
- [x] Syntax validation passed
- [x] Fields have correct defaults
- [x] Fields follow existing patterns
