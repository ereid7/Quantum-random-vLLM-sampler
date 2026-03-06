"""M1: Logit noise injection method."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from qr_sampler.exceptions import EntropyUnavailableError

if TYPE_CHECKING:
    from qr_sampler.config import QRSamplerConfig
    from qr_sampler.entropy.base import EntropySource

_logger = logging.getLogger("qr_sampler")
_SQRT2 = math.sqrt(2.0)


class LogitNoise:
    """M1: Gaussian logit noise injection.

    Adds quantum-seeded Gaussian noise to logits before temperature
    scaling, reshaping the probability distribution at the earliest
    pipeline stage.

    Design note: One quantum-derived seed drives a numpy Generator that
    produces N Gaussian samples. This 1:N dilution is intentional \u2014
    correlated noise preserves token relevance better than independent
    per-logit noise (confirmed by M1 experiments in quantum-llama.cpp).
    """

    @staticmethod
    def perturb(
        logits: np.ndarray[Any, np.dtype[np.floating[Any]]],
        entropy_source: EntropySource,
        config: QRSamplerConfig,
    ) -> np.ndarray[Any, np.dtype[np.floating[Any]]]:
        """Perturb logits with quantum-seeded Gaussian noise.

        Args:
            logits: 1-D float array of raw logit values.
            entropy_source: Source of quantum entropy bytes.
            config: Sampler configuration (uses logit_noise_alpha, logit_noise_sigma,
                sample_count, population_mean, population_std, injection_verbose).

        Returns:
            Modified logits array (same shape as input). Returns input unchanged
            if logit_noise_alpha == 0 or entropy is unavailable.
        """
        if config.logit_noise_alpha == 0.0:
            return logits

        try:
            raw_bytes = entropy_source.get_random_bytes(config.sample_count)
        except EntropyUnavailableError:
            _logger.warning("M1 LogitNoise: entropy unavailable, skipping perturbation")
            return logits

        if not raw_bytes:
            _logger.warning("M1 LogitNoise: empty entropy payload, skipping perturbation")
            return logits

        # Z-score \u2192 normal CDF \u2192 uniform value (same math as ZScoreMeanAmplifier)
        samples = np.frombuffer(raw_bytes, dtype=np.uint8)
        n = len(samples)
        sample_mean = float(np.mean(samples))
        sem = config.population_std / math.sqrt(n)
        z_score = (sample_mean - config.population_mean) / sem
        u = 0.5 * (1.0 + math.erf(z_score / _SQRT2))
        eps = config.uniform_clamp_epsilon
        u = max(eps, min(1.0 - eps, u))

        # Seed numpy Generator from quantum-derived u value.
        # One seed \u2192 N correlated Gaussian samples (intentional 1:N dilution).
        rng = np.random.default_rng(int(u * 2**32))
        noise = rng.standard_normal(len(logits)) * config.logit_noise_sigma
        result = logits + config.logit_noise_alpha * noise

        if config.injection_verbose:
            _logger.debug(
                "M1 LogitNoise: alpha=%.4f sigma=%.4f u=%.6f",
                config.logit_noise_alpha,
                config.logit_noise_sigma,
                u,
            )

        return result
