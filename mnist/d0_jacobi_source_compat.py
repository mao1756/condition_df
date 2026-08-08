"""Reviewed additive-source successors for immutable Jacobi provenance.

Historical Jacobi runs deliberately bind the source bytes that produced them.
The one-image learnability patch adds two non-mutating capabilities:

* a read-only capture payload on the exact multipath scheduler; and
* source-fingerprint support for distinguishing that additive successor from
  the historical scientific core.

The old provenance view first recognizes the exact LF/CRLF artifact-helper
successor, then accepts only a finite allowlist of complete source-set
fingerprints.  It never rewrites the scheduler by basename.  Any additional,
missing, or changed source therefore fails closed.  New learnability
manifests do not apply this table and bind all current bytes directly.
"""

from __future__ import annotations

from typing import Final


HISTORICAL_ARTIFACT_HELPER_SUCCESSORS: Final[dict[str, str]] = {
    "affa496821dcfda8a5e58273f565ccd2703f63115f2bf90e325382a1daeda82f": (
        "fd261a05b209d80e7abb43c43fdae713f1b0a7058f12187b53731f9e5fe516b5"
    ),
    "946cc6002be7f88118128f3cab885485ed7f351ed0777440cf6377d56198e5c4": (
        "fd261a05b209d80e7abb43c43fdae713f1b0a7058f12187b53731f9e5fe516b5"
    ),
}

HISTORICAL_SOURCE_SET_SUCCESSORS: Final[
    dict[tuple[str, str], str]
] = {
    (
        "093cf60b184657148f8bdc536e0d1388a1d9b84b3f924103b43b3f25529101aa",
        "4b87dfea2e55cb6674eb8fbf7aa9f3dc5fabb5eb08c9e02627cdc5e26bc51cfc",
    ): "300bcdab17d9cac5605311bf0b513a5c476e88011662fb1e51ac69ca4f431c39",
    (
        "22ad69e7ba7784f75af9f65f36ce1e18c02eb88e23e34c513df7746e22ae0520",
        "a096cbd1401b8918a6222611b67a52c2416e8c85efb15c5a02478c84eb3f25e5",
    ): "19086844e84f141c8aa86a235746ef5ef1376c6d1365253e3045d31df9524e6a",
    (
        "353ab53aa4dbf2b7498233c7a346b1f5e9cf27e22c67b1000b5db3317f7630f6",
        "a38cb5fc5e9e6c164d369eb156f40e378cd2fb0e38c74039ee7b7a3725a32196",
    ): "a38cb5fc5e9e6c164d369eb156f40e378cd2fb0e38c74039ee7b7a3725a32196",
    (
        "44de20a3f5e0259737a25adadee00d612d71f760b425fd20ea411e259a4969c6",
        "c8f6524b36477782df49eb75e4fcfe26d43151f16ae7e1bbb7560dadb1a2488e",
    ): "ff0b17a9efb5c321fe3c9cc23e8ac5382f3c51f742d1b329fff97fdcd29685ec",
    (
        "525696341c67dcb82c08eaaa92983f427a5aff44891d7e040edacf1c1cced7da",
        "a07df322db69b53507c29a70a3dbc5b2487e36b043d2076520751af852f7ddc5",
    ): "151eaa6c3fbd3a4beaae61ad5337892187e4338fe629761716f281bb84f7d450",
    (
        "58b9b76873612b2cb8c688c4296f46d620588b44a33fba6e9cc216259276b2a8",
        "e7803a1e9fac0719e2a51aa069cb6a0c52fdae6eca049bd593501ac96ba30031",
    ): "e7803a1e9fac0719e2a51aa069cb6a0c52fdae6eca049bd593501ac96ba30031",
    (
        "6bcfd31e21a7d26e88057274388e5731175023d9e6344a865a07b3ebb6156766",
        "2c68149ec1a3cf658b4b09bd55aeda41cedb44bf89162b3c826b6bbbdab811d1",
    ): "5fcf9af561f40c0f7dd0f41c4b886fe332b46e65c8dce993a5850f534a7b1a9e",
    (
        "f975eeacafe34333e3a29aaf1b6fad575f1972d0c320183deac3416597d89925",
        "98cc49304cc45885bda8f6f0efb7852fa757b25b44298eaa8dff0efc4698cd9c",
    ): "2f20297eb83b434aa782676119915e9f8883eb116cec0d2b08c2c8c9a8b5ddb0",
    # Exact CRLF form of the reviewed scheduler successor.
    (
        "093cf60b184657148f8bdc536e0d1388a1d9b84b3f924103b43b3f25529101aa",
        "61eb56baf78fc6a4c9ef83e70a44342fb574b6658d51c6c38181718d2b193ec6",
    ): "300bcdab17d9cac5605311bf0b513a5c476e88011662fb1e51ac69ca4f431c39",
    (
        "22ad69e7ba7784f75af9f65f36ce1e18c02eb88e23e34c513df7746e22ae0520",
        "bba3f3a7b3761e48e4b06701998b53b35904b97ff59c1c6bce59aa1d0a56660c",
    ): "19086844e84f141c8aa86a235746ef5ef1376c6d1365253e3045d31df9524e6a",
    (
        "44de20a3f5e0259737a25adadee00d612d71f760b425fd20ea411e259a4969c6",
        "b1ad700b4b8074c32b3b7ceecf9d4041cef4c42f7271fcdd42fbef288b2d999a",
    ): "ff0b17a9efb5c321fe3c9cc23e8ac5382f3c51f742d1b329fff97fdcd29685ec",
    (
        "525696341c67dcb82c08eaaa92983f427a5aff44891d7e040edacf1c1cced7da",
        "c1e3e02791ecab3af30121de3c9173509971dab851bce93574804ea5d9fd92f5",
    ): "151eaa6c3fbd3a4beaae61ad5337892187e4338fe629761716f281bb84f7d450",
    (
        "6bcfd31e21a7d26e88057274388e5731175023d9e6344a865a07b3ebb6156766",
        "c1351867e5beabeb9b51ceef3ae69626c24fee859a5f2d226c143ae5a27ecb55",
    ): "5fcf9af561f40c0f7dd0f41c4b886fe332b46e65c8dce993a5850f534a7b1a9e",
    (
        "f975eeacafe34333e3a29aaf1b6fad575f1972d0c320183deac3416597d89925",
        "1776f02ae56759ea054097b7dcb034ae08647ccc71086c292b8a82f6591fe1f9",
    ): "2f20297eb83b434aa782676119915e9f8883eb116cec0d2b08c2c8c9a8b5ddb0",
}

__all__ = [
    "HISTORICAL_ARTIFACT_HELPER_SUCCESSORS",
    "HISTORICAL_SOURCE_SET_SUCCESSORS",
]
