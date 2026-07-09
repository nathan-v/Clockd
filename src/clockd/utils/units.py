def ms_to_mph(ms: float) -> float:
    return ms * 2.23694


def ms_to_kmh(ms: float) -> float:
    return ms * 3.6


def mph_to_ms(mph: float) -> float:
    return mph / 2.23694


def convert_speed(ms: float, unit: str) -> float:
    if unit == "kmh":
        return round(ms_to_kmh(ms), 1)
    return round(ms_to_mph(ms), 1)
