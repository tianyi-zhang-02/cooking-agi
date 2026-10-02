import argparse
import json
from pathlib import Path


VERDICTS = {"pass", "fail", "unknown"}
OUTPUT_FIELDS = {"criterion_id", "verdict", "evidence_ids", "reason"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_judgment(raw, evidence_ids, criterion_id):
    if raw is None:
        return {"status": "error", "verdict": None, "detail": "No response"}
    try:
        if not isinstance(raw, str):
            raise ValueError("Response must be a JSON string")
        payload = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(payload, dict) or set(payload) != OUTPUT_FIELDS:
            raise ValueError("Unexpected output fields")
        if payload["criterion_id"] != criterion_id:
            raise ValueError("Wrong criterion version")
        if not isinstance(payload["verdict"], str) or payload["verdict"] not in VERDICTS:
            raise ValueError("Unknown verdict")
        reason = payload["reason"]
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 600:
            raise ValueError("Expected a short, nonempty reason")
        citations = payload["evidence_ids"]
        if not isinstance(citations, list) or any(not isinstance(value, str) for value in citations):
            raise ValueError("Evidence IDs must be strings")
        if len(citations) != len(set(citations)) or not set(citations).issubset(evidence_ids):
            raise ValueError("Duplicate or unknown evidence IDs")
        if payload["verdict"] != "unknown" and not citations:
            raise ValueError("This grounding contract requires evidence for a decision")
    except (ValueError, TypeError) as error:
        return {"status": "invalid", "verdict": None, "detail": str(error)}
    return {
        "status": "unknown" if payload["verdict"] == "unknown" else "decided",
        "verdict": payload["verdict"],
        "detail": reason,
    }


def validate_cases(records):
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each case must be an object")
        for field in ("case_id", "group_id", "slice"):
            if not isinstance(record.get(field), str) or not record[field].strip():
                raise ValueError(f"Missing {field}")
        if record["case_id"] in seen:
            raise ValueError("Duplicate case_id")
        seen.add(record["case_id"])
        if record.get("human") not in ("pass", "fail"):
            raise ValueError("The demo requires human pass/fail labels")
        identifiers = record.get("allowed_evidence_ids")
        if not isinstance(identifiers, list) or any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("Invalid evidence allowlist")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate allowed evidence ID")
        if "judge_raw" not in record:
            raise ValueError("Missing judge_raw; use null for a failed call")


def summarize(evaluated):
    counts = {status: 0 for status in ("decided", "unknown", "invalid", "error")}
    matrix = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    for record in evaluated:
        counts[record["status"]] += 1
        if record["status"] != "decided":
            continue
        if record["verdict"] == "pass":
            key = "tp" if record["human"] == "pass" else "fp"
        else:
            key = "fn" if record["human"] == "pass" else "tn"
        matrix[key] += 1
    total = len(evaluated)
    approvals = matrix["tp"] + matrix["fp"]
    return {
        "total_cases": total,
        **counts,
        "coverage": counts["decided"] / total if total else None,
        "accuracy_on_decided": (matrix["tp"] + matrix["tn"]) / counts["decided"] if counts["decided"] else None,
        "error_among_approvals": matrix["fp"] / approvals if approvals else None,
        "confusion_matrix": matrix,
    }


def evaluate(records, criterion_id="policy-grounding-v1"):
    validate_cases(records)
    evaluated = []
    for record in records:
        result = parse_judgment(record["judge_raw"], set(record["allowed_evidence_ids"]), criterion_id)
        evaluated.append({
            "case_id": record["case_id"],
            "group_id": record["group_id"],
            "slice": record["slice"],
            "human": record["human"],
            **result,
        })
    return {
        "notice": "Fictional fixture; not measured model performance. No model is called.",
        "criterion_id": criterion_id,
        "overall": summarize(evaluated),
        "slices": {
            label: summarize([record for record in evaluated if record["slice"] == label])
            for label in sorted({record["slice"] for record in evaluated})
        },
        "cases": evaluated,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate fictional judge outputs without calling a model.")
    parser.add_argument("path", nargs="?", type=Path, default=Path(__file__).with_name("example-results.jsonl"))
    args = parser.parse_args()
    try:
        with args.path.open(encoding="utf-8") as source:
            records = [json.loads(line, object_pairs_hook=unique_object) for line in source if line.strip()]
        report = evaluate(records)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
