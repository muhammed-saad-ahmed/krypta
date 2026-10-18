from config import RISK_WEIGHTS, RISK_LEVELS, RAIN_SENSITIVE, ROAD_HEALTH_WEIGHTS

def _clip(x): return max(0.0, min(1.0, float(x)))

def evidence_confidence(observations):
    """Noisy-OR over INDEPENDENT sources (max AI confidence per user/source).
    Repeat frames from one source do not inflate confidence."""
    best = {}
    for o in observations:
        k = o.get("source_id", "unknown")
        best[k] = max(best.get(k, 0.0), _clip(o.get("confidence", 0)))
    p = 1.0
    for c in best.values(): p *= (1 - c * 0.85)  # 0.85 caps trust in any one source
    return round(1 - p, 4) if best else 0.0

def risk_score(inc, weights=None, rainfall_mm=0.0):
    """Transparent weighted score in [0,100] with reasons and missing-data notes."""
    w = weights or RISK_WEIGHTS
    total = sum(w.values()) or 1.0
    missing = []
    exposure = inc.get("traffic_exposure")
    if exposure is None: exposure = 0.5; missing.append("traffic exposure unknown (neutral 0.5 used)")
    n = int(inc.get("n_observations", 1))
    rain = _clip(rainfall_mm / 100.0) * RAIN_SENSITIVE.get(inc.get("type"), 0.2)
    comps = {"severity": _clip(inc.get("severity", 0.5)), "exposure": _clip(exposure),
             "recurrence": _clip((n - 1) / 4.0), "environment": rain,
             "confirmation": _clip(inc.get("evidence_confidence", 0.0))}
    score = 100 * sum(w[k] * v for k, v in comps.items()) / total
    level = next(l for t, l in RISK_LEVELS if score >= t)
    names = {"severity": "high hazard severity", "exposure": "high traffic exposure",
             "recurrence": "repeated observations", "environment": "adverse weather context",
             "confirmation": "multi-source confirmation"}
    reasons = [names[k] for k, v in comps.items() if v >= 0.6]
    if n < 2: missing.append("only one observation available")
    conf = "Low" if len(missing) >= 2 else "Medium" if missing else "High"
    return {"score": round(score, 1), "level": level, "components": comps, "reasons": reasons,
            "risk_confidence": conf, "uncertainty_notes": missing,
            "disclaimer": "Experimental research score; not an official safety rating."}

def road_health(components, weights=None):
    """components: 0-100 per factor (higher = healthier)."""
    w = weights or ROAD_HEALTH_WEIGHTS
    t = sum(w.values()) or 1.0
    return round(sum(w[k] * components.get(k, 50) for k in w) / t, 1)
