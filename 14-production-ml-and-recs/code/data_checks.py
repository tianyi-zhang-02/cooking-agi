import json
import math


FEATURES = [
    {"event": 590, "available": 591, "value": 3},
    {"event": 598, "available": 604, "value": 7},
    {"event": 602, "available": 602, "value": 9},
]


def feature_at(records, decision_time, ttl_minutes):
    if not math.isfinite(decision_time) or not math.isfinite(ttl_minutes) or ttl_minutes < 0:
        raise ValueError("Decision time must be finite and TTL nonnegative")
    eligible = []
    for record in records:
        if not all(math.isfinite(record[field]) for field in ("event", "available")):
            raise ValueError("Feature timestamps must be finite")
        if (decision_time - ttl_minutes <= record["event"] <= decision_time
                and record["available"] <= decision_time):
            eligible.append(record)
    if not eligible:
        return None
    selected = max(eligible, key=lambda record: (record["event"], record["available"]))
    return selected["value"]


def original_probability(sampled_probability, negative_keep_probability):
    if not 0 <= sampled_probability <= 1 or not 0 < negative_keep_probability <= 1:
        raise ValueError("Require a valid probability and a positive negative-sampling rate")
    retained = negative_keep_probability * sampled_probability
    return retained / (1 - sampled_probability + retained)


def click_rate(clicks, impressions):
    if not (type(clicks) is int and type(impressions) is int
            and 0 <= clicks <= impressions and impressions > 0):
        raise ValueError("Require integer counts with 0 <= clicks <= positive impressions")
    return clicks / impressions


def example_results():
    return {
        "available_feature": feature_at(FEATURES, 600, 60),
        "event_time_only_feature": max(
            (record for record in FEATURES if record["event"] <= 600),
            key=lambda record: record["event"],
        )["value"],
        "corrected_probability": original_probability(0.5, 0.1),
        "overall_ctr": [click_rate(1000, 20000), click_rate(1200, 20000)],
        "new_user_ctr": [click_rate(100, 2000), click_rate(60, 2000)],
    }


if __name__ == "__main__":
    print(json.dumps(example_results(), indent=2))
