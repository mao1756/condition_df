"""Reviewed source successors introduced by the boundary-tangent v3 patch.

This table lives outside every historical source closure.  That placement is
intentional: it lets the current artifact helper recognize exact additive
successors without making the compatibility table's own bytes part of the
historical aggregate being normalized.
"""

from __future__ import annotations

from typing import Final


V3_ARTIFACT_HELPER_SUCCESSORS: Final[dict[str, str]] = {
    "2e2f2447d2e2bb711a6cc271bcce2d2e40318006eac92973a0e6401436443451": (
        "affa496821dcfda8a5e58273f565ccd2703f63115f2bf90e325382a1daeda82f"
    ),
    "68f70a2f14ec41ee9fbd5ac2f9994591f87c75e4176de7eb7694dc15495335b5": (
        "affa496821dcfda8a5e58273f565ccd2703f63115f2bf90e325382a1daeda82f"
    ),
    "75bac4e947349993f6f7bdc3cf6df31e0861f67d8fbe7688d3761ee7d6325e21": (
        "affa496821dcfda8a5e58273f565ccd2703f63115f2bf90e325382a1daeda82f"
    ),
}

V3_HISTORICAL_SOURCE_SET_SUCCESSORS: Final[
    dict[tuple[str, str], str]
] = {
    (
        "dd60fe4b2145902a2d72b7888bca13d96a743d569fbbb5f9f7d672714f576975",
        "1b2291dc035ac9f268b8e8ef0bbad5fe2c0d5eba37a006928e2085f7f271f298",
    ): "c562b8e39a07bbc19a9f65b9a3187c49d97cb64a277bf9492f4cb7fb92c9b2ee",
    (
        "7a2c8f5d432e6cd34b293a9e9b750f71d80bf81d5189114e6c49b5c573b27e8d",
        "ae85c4c898ec04e8abf0ad61e765781cabdfea36fc11e267422a3e4af9af4236",
    ): "e30e301ffa330108c986dfb80f32ac2d4d17f648b422a9b047ef5e807e826547",
    (
        "a778458357effec4ef7e7b0388768634532db2469096919586e901f4437fe490",
        "886175c803fd08618183fe12628af4b9341daad7bdca395e3ad77ffd4b0bc1b9",
    ): "50e9ebef34ed7981c8dc3a3a72e8c39c75eda162898f6d6648c7a618d85e5b87",
    (
        "bd4f3bb94b044a0047d454ab80a7da11b945bdd36dfdf39dac1ead6ce810f31a",
        "17c402e9009fbbbc8ba3bb18591dc00c29e60e1554ef38f870a3fff781889fe9",
    ): "42b7129b8850d5e4036137e1781799fe0d9b37d8ee98867d3e1a7b7a57b7906c",
    (
        "676e4aea5a866f5a06a21b408b8b787e5169c66110798773700d550dbf499e34",
        "96822971f35246bfc70a96c68a77f33b63c51437e3beebfe5adbbe4e8abf171e",
    ): "54cfa6896de2ce7da3cd4190a01d113a04aee328dad0bc62e7d6a8f1aaa3a215",
}


__all__ = [
    "V3_ARTIFACT_HELPER_SUCCESSORS",
    "V3_HISTORICAL_SOURCE_SET_SUCCESSORS",
]
