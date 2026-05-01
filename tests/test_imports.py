"""Smoke checks for the direct package imports.

Run with
    .venv\\Scripts\\python.exe -m tests.test_imports
"""

from __future__ import annotations

import core.conditioning_utils as conditioning_utils
import core.wasserstein_conditioning_algorithms as wasserstein_conditioning_algorithms
import mnist.conditioned_diffusion as mnist_conditioned_diffusion
import mnist.experiment6_fixes as mnist_experiment6_fixes
import mnist.experiment6_hyperparameter_search as mnist_experiment6_hyperparameter_search
import mnist.score_matching as mnist_score_matching
import mnist.weighted_point_cloud as mnist_weighted_point_cloud


def _assert_alias(module: object, expected_name: str) -> None:
    actual = getattr(module, "__name__", "")
    if actual != expected_name:
        raise AssertionError(f"expected alias to {expected_name!r}, got {actual!r}")


def test_direct_package_imports() -> None:
    _assert_alias(conditioning_utils, "core.conditioning_utils")
    _assert_alias(
        wasserstein_conditioning_algorithms,
        "core.wasserstein_conditioning_algorithms",
    )
    _assert_alias(mnist_weighted_point_cloud, "mnist.weighted_point_cloud")
    _assert_alias(mnist_conditioned_diffusion, "mnist.conditioned_diffusion")
    _assert_alias(mnist_experiment6_fixes, "mnist.experiment6_fixes")
    _assert_alias(
        mnist_experiment6_hyperparameter_search,
        "mnist.experiment6_hyperparameter_search",
    )
    _assert_alias(mnist_score_matching, "mnist.score_matching")

    assert hasattr(mnist_score_matching, "train_score_model")
    assert hasattr(mnist_conditioned_diffusion, "generate_guided_point_clouds")
    assert hasattr(wasserstein_conditioning_algorithms, "simulate_wasserstein_mc_sinkhorn_em")


def main() -> None:
    test_direct_package_imports()

    print("test_imports: OK")


if __name__ == "__main__":
    main()
