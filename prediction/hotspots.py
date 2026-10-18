"""
Recurring-hotspot flagging: NOT a trained forecasting model. It flags locations with enough
repeated observations to call "recurring" rather than predicting anything about the future.
Returns None (distinct from an empty list) when there isn't enough data to say anything —
callers should render "Prediction unavailable — additional historical data required." for None.
"""
MIN_OBSERVATIONS_FOR_PREDICTION = 3


def recurring_hotspots(df, min_observations=MIN_OBSERVATIONS_FOR_PREDICTION):
    """df needs an 'n_observations' column (and ideally 'type', 'risk', 'code'). Returns the
    subset with n_observations >= min_observations, sorted by observation count descending,
    or None if there's no usable data or nothing qualifies."""
    if df is None or len(df) == 0 or "n_observations" not in getattr(df, "columns", []):
        return None
    candidates = df[df["n_observations"] >= min_observations]
    if len(candidates) == 0:
        return None
    return candidates.sort_values("n_observations", ascending=False)
