"""Print the recommended XH-202615 data source plan."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    path = Path("configs/data_sources.json")
    plan = json.loads(path.read_text(encoding="utf-8"))
    print("Rules")
    for key, value in plan["rules"].items():
        print(f"- {key}: {value}")
    for priority in ["priority_0", "priority_1", "priority_2"]:
        print(f"\n{priority}")
        for item in plan[priority]:
            roles = ", ".join(item["role"])
            print(f"- {item['name']}: {roles}")
            print(f"  {item['url']}")


if __name__ == "__main__":
    main()

