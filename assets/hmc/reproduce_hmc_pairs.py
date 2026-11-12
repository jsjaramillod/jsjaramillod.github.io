"""Estimate stationary two-draw variance ratios for exact HMC on the circle.

Run with Python 3 and NumPy: python reproduce_hmc_pairs.py
Each independent replicate draws a von Mises position and a Laplace momentum,
then advances the exact Hamiltonian dynamics by the specified travel length.
"""
import numpy as np


KAPPA = 4.0
LENGTH = 1.4
SEED = 20261112
BATCH_COUNT = 120
BATCH_SIZE = 10000


def wrap(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi


def hmc_step(theta, momentum, kappa, length):
    """Unit-scale Laplace kinetic energy |p|, potential -kappa*cos(theta)."""
    energy = -kappa * np.cos(theta) + np.abs(momentum)
    unfolded = theta + np.sign(momentum) * length
    endpoint = wrap(unfolded)
    trapped = energy < kappa
    # Motions below the barrier reflect at the turning points -A and A.
    angle = np.arccos(np.clip(-energy[trapped] / kappa, -1.0, 1.0))
    if np.any(angle == 0):
        raise ValueError("Zero-energy-width orbit; this has probability zero.")
    phase = (unfolded[trapped] + angle) % (4 * angle)
    endpoint[trapped] = angle - np.abs(phase - 2 * angle)
    return endpoint


def estimate_ratio(values0, values1):
    """Var[(f0+f1)/2] divided by Var_pi[f]/2, pooling both marginals."""
    mean = 0.5 * (np.mean(values0) + np.mean(values1))
    left = values0 - mean
    right = values1 - mean
    stationary_variance = 0.5 * (np.mean(left**2) + np.mean(right**2))
    pair_variance = np.mean(((left + right) / 2)**2)
    return pair_variance / (stationary_variance / 2)


def reference_moments(observable):
    """Periodic quadrature of the stationary von Mises mean and variance."""
    grid = np.linspace(-np.pi, np.pi, 16384, endpoint=False)
    weights = np.exp(KAPPA * np.cos(grid))
    values = observable(grid)
    mean = np.average(values, weights=weights)
    return mean, np.average((values - mean)**2, weights=weights)


def main():
    rng = np.random.default_rng(SEED)
    count = BATCH_SIZE * BATCH_COUNT
    theta = rng.vonmises(0, KAPPA, count)
    momentum = rng.laplace(size=count)
    endpoint = hmc_step(theta, momentum, KAPPA, LENGTH)
    identity = hmc_step(theta, momentum, KAPPA, 0)
    assert np.max(np.abs(wrap(identity - theta))) < 1e-12
    print(f"{count:,} independent pairs; seed {SEED}; kappa={KAPPA}; length={LENGTH}")
    for observable in (np.sin, np.cos):
        left = observable(theta)
        right = observable(endpoint)
        ratio = estimate_ratio(left, right)
        batch_ratios = [
            estimate_ratio(x, y)
            for x, y in zip(left.reshape(BATCH_COUNT, -1), right.reshape(BATCH_COUNT, -1))
        ]
        mcse = np.std(batch_ratios, ddof=1) / np.sqrt(BATCH_COUNT)
        target_mean, target_var = reference_moments(observable)
        print(f"{observable.__name__}: ratio={ratio:.6f}, MCSE={mcse:.6f}, "
              f"approx. 95% MC interval=[{ratio - 1.96 * mcse:.6f}, {ratio + 1.96 * mcse:.6f}]")
        print(f"  means (initial, endpoint, target): "
              f"{left.mean():.6f}, {right.mean():.6f}, {target_mean:.6f}")
        print(f"  variances (initial, endpoint, target): "
              f"{left.var():.6f}, {right.var():.6f}, {target_var:.6f}")


if __name__ == "__main__":
    main()
