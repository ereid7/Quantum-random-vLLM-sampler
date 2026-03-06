"""M3: Correlated walk injection method."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from qr_sampler.exceptions import EntropyUnavailableError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource

_logger = logging.getLogger("qr_sampler")
_SQRT2 = math.sqrt(2.0)


class CorrelatedWalk:
    """M3: Quantum correlated walk.

    Maintains a per-request walk position that drifts based on quantum
    entropy, replacing the amplified u value with the walk position.
    This creates temporal correlations across tokens within a request.

    The walk position evolves as:
        new_position = (walk_position + step * (qval - 0.5)) % 1.0
    where qval is the quantum-derived uniform value in (0, 1).
    The modulo wraps the position to stay in [0, 1).
    """

    @staticmethod
    def step(
        u: float,
        entropy_source: EntropySource,
        config: QRSamplerConfig,
        walk_position: float,
    ) -> tuple[float, float]:
        """Advance the walk by one step and return the new u value.

        Args:
            u: Current amplified uniform value from the signal amplifier.
            entropy_source: Source of quantum entropy bytes.
            config: Sampler configuration (uses walk_step, sample_count,
                population_mean, population_std, injection_verbose).
            walk_position: Current walk position in [0, 1).

        Returns:
            Tuple of (new_u, new_walk_position). Both values are the new
            walk position. Returns (u, walk_position) unchanged if
            walk_step == 0 or entropy is unavailable.
        """
        if config.walk_step == 0.0:
            return (u, walk_position)

        try:
            raw_bytes = entropy_source.get_random_bytes(config.sample_count)
        except EntropyUnavailableError:
            _logger.warning("M3 CorrelatedWalk: entropy unavailable, skipping step")
            return (u, walk_position)

        if not raw_bytes:
            _logger.warning("M3 CorrelatedWalk: empty entropy payload, skipping step")
            return (u, walk_position)

        # Z-score \u2192 normal CDF \u2192 uniform value (same math as ZScoreMeanAmplifier)
        samples = np.frombuffer(raw_bytes, dtype=np.uint8)
        n = len(samples)
        sample_mean = float(np.mean(samples))
        sem = config.population_std / math.sqrt(n)
        z_score = (sample_mean - config.population_mean) / sem
        qval = 0.5 * (1.0 + math.erf(z_score / _SQRT2))
        eps = config.uniform_clamp_epsilon
        qval = max(eps, min(1.0 - eps, qval))

        # Update walk position: drift by step * (qval - 0.5)
        # qval in (0,1) \u2192 (qval - 0.5) in (-0.5, 0.5) \u2192 drift in (-step/2, step/2)
        new_position = walk_position + config.walk_step * (qval - 0.5)

        # Wrap to [0, 1) using modulo \u2014 Python's % handles negatives correctly:
        # e.g., -0.1 % 1.0 == 0.9
        new_position = new_position % 1.0

        if config.injection_verbose:
            _logger.debug(
                "M3 CorrelatedWalk: step=%.4f qval=%.6f old_pos=%.6f new_pos=%.6f",
                config.walk_step,
                qval,
                walk_position,
                new_position,
            )

        return (new_position, new_position)
