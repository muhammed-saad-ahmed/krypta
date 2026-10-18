import math
from config import DUP_RADIUS_M

def haversine_m(a, b):
    R = 6371000.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(h))

def valid_coord(lat, lon, bbox=None):
    try: lat, lon = float(lat), float(lon)
    except (TypeError, ValueError): return False
    if not (-90 <= lat <= 90 and -180 <= lon <= 180): return False
    if bbox: return bbox[0] <= lat <= bbox[2] and bbox[1] <= lon <= bbox[3]
    return True

def cluster_observations(obs, radius_m=DUP_RADIUS_M):
    """Merge same-type observations within radius into one incident (greedy, centroid-based).
    Returns list of incidents with member indices and merge reasons."""
    clusters = []
    for i, o in enumerate(obs):
        if not valid_coord(o.get("lat"), o.get("lon")): continue
        for c in clusters:
            d = haversine_m((c["lat"], c["lon"]), (o["lat"], o["lon"]))
            if c["type"] == o["type"] and d <= radius_m:
                c["members"].append(i); c["reasons"].append(f"obs {i}: same type, {d:.1f} m from centroid")
                n = len(c["members"])
                c["lat"] += (o["lat"] - c["lat"]) / n; c["lon"] += (o["lon"] - c["lon"]) / n
                break
        else:
            clusters.append({"type": o["type"], "lat": o["lat"], "lon": o["lon"],
                             "members": [i], "reasons": []})
    return clusters
