"""M2: Temperature variance injection method."""

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
_MIN_TEMPERATURE = 0.01


class TempVariance:
    """M2: Quantum temperature modulation.

    Modulates the computed temperature value using quantum entropy,
    introducing stochastic variation in the sharpness of the probability
    distribution.

    Formula: new_temp = temperature * (1 + beta * (u - 0.5))
    where u is the quantum-derived uniform value in (0, 1).
    Result is clamped to [0.01, inf) to prevent degenerate distributions.
    """

    @staticmethod
    def modulate(
        temperature: float,
        entropy_source: EntropySource,
        config: QRSamplerConfig,
    ) -> float:
        """Modulate temperature with quantum entropy.

        Args:
            temperature: Base temperature value from the temperature strategy.
            entropy_source: Source of quantum entropy bytes.
            config: Sampler configuration (uses temp_variance_beta, sample_count,
                population_mean, population_std, injection_verbose).

        Returns:
            Modulated temperature, clamped to >= 0.01. Returns input unchanged
            if temp_variance_beta == 0 or entropy is unavailable.
        """
        if config.temp_variance_beta == 0.0:
            return temperature

        try:
            raw_bytes = entropy_source.get_random_bytes(config.sample_count)
        except EntropyUnavailableError:
            _logger.warning("M2 TempVariance: entropy unavailable, skipping modulation")
            return temperature

        if not raw_bytes:
            _logger.warning("M2 TempVariance: empty entropy payload, skipping modulation")
            return temperature

        # Z-score \u2192 normal CDF \u2192 uniform value (same math as ZScoreMeanAmplifier)
        samples = np.frombuffer(raw_bytes, dtype=np.uint8)
        n = len(samples)
        sample_mean = float(np.mean(samples))
        sem = config.population_std / math.sqrt(n)
        z_score = (sample_mean - config.population_mean) / sem
        u = 0.5 * (1.0 + math.erf(z_score / _SQRT2))
        eps = config.uniform_clamp_epsilon
        u = max(eps, min(1.0 - eps, u))

        # Modulate: scale temperature by (1 + beta * (u - 0.5))
        # u in (0,1) \u2192 (u - 0.5) in (-0.5, 0.5) \u2192 modulation in (-beta/2, beta/2)
        modulation = config.temp_variance_beta * (u - 0.5)
        new_temp = temperature * (1.0 + modulation)
        new_temp = max(_MIN_TEMPERATURE, new_temp)

        if config.injection_verbose:
            _logger.debug(
                "M2 TempVariance: beta=%.4f u=%.6f original=%.4f new=%.4f",
                config.temp_variance_beta,
                u,
                temperature,
                new_temp,
            )

        return new_temp
