"""Maritime Identification Digits (MID): first 3 digits of a vessel MMSI (ITU)."""


def mmsi_to_mid(mmsi: int) -> int:
    """
    Return the 3-digit country/region code encoded in the MMSI.

    Standard 9-digit MMSI: MID = floor(MMSI / 1_000_000). Some datasets omit
    leading zeros; we treat values >= 100_000_000 as 9-digit. Smaller values
    fall back to the legacy heuristic used in early project scripts.
    """
    m = int(mmsi)
    if m < 0:
        m = -m
    if m >= 100_000_000:
        return m // 1_000_000
    if m >= 10_000_000:
        return m // 1_000_000
    return m // 1_000
