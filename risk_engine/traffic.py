import config


def density_level(vehicle_count):
    """Maps a raw vehicle count to a density band using config.TRAFFIC_DENSITY_THRESHOLDS.
    These bands are configured for this prototype, not calibrated against measured real-world
    traffic flow — treat the label as an experimental estimate, not a traffic-engineering figure."""
    for label, (lo, hi) in config.TRAFFIC_DENSITY_THRESHOLDS.items():
        if lo <= vehicle_count <= hi:
            return label
    return list(config.TRAFFIC_DENSITY_THRESHOLDS)[-1]
