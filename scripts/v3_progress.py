"""Read-only progress for all v3 conditions; never treats partial scores as final."""
from datetime import datetime
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    conditions = []
    for i in range(1, 5):
        directory = root/f"artifacts/repar_v3_20261119_v{i}"
        result = {"condition": i, "exists": directory.exists()}
        for key, relative, expected in (("math_predictions", "math/predictions.jsonl", 19),
                                        ("math_grades", "math/grades.jsonl", 19),
                                        ("longbench_predictions", "longbench/predictions.jsonl", 125)):
            path = directory/relative
            values = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()] if path.exists() else []
            result[key] = {"rows": len(values), "expected": expected,
                           "unique": len({v["id"] for v in values}),
                           "errors": sum(v.get("status") == "error" for v in values)}
        result["summary_complete"] = (directory/"summary.json").exists()
        result["audit_passed"] = json.loads((directory/"audit.json").read_text(encoding="utf-8")).get("passed") if (directory/"audit.json").exists() else False
        conditions.append(result)
    print(json.dumps({"time": datetime.now().astimezone().isoformat(), "conditions": conditions}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
