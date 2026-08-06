import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class Phase1ReplayCliTest(unittest.TestCase):
    def test_replay_runner_writes_provenance_outputs_without_labels(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "pos.jsonl").write_text(
                json.dumps(
                    {
                        "id": "p1",
                        "wakeup_audio": "missing-wakeup.wav",
                        "wakeup_text": "小助手",
                        "command_audio": "p1.wav",
                        "recognition_text": "打开空调",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (dataset / "neg.jsonl").write_text(
                json.dumps(
                    {
                        "id": "n1",
                        "wakeup_audio": "missing-wakeup.wav",
                        "wakeup_text": "小助手",
                        "command_audio": "n1.wav",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            asr_map = root / "asr.jsonl"
            asr_map.write_text(
                "\n".join(
                    [
                        json.dumps({"id": "p1", "recognition_text": "打开空调"}, ensure_ascii=False),
                        json.dumps({"id": "n1", "recognition_text": "背景声"}, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            speaker_scores = root / "speaker.csv"
            speaker_scores.write_text(
                "id,target_probability,overlap_probability,global_similarity\n"
                "p1,1.0,0.0,0.8\n",
                encoding="utf-8",
            )
            outputs = {name: root / name for name in ("predictions.jsonl", "evidence.jsonl", "routes.jsonl", "trace.json")}
            command = [
                sys.executable,
                str(repo / "scripts" / "run_phase1_replay.py"),
                "--dataset-root",
                str(dataset),
                "--asr-map",
                str(asr_map),
                "--speaker-scores",
                str(speaker_scores),
                "--config",
                str(repo / "configs" / "phase1_replay.json"),
                "--predictions-out",
                str(outputs["predictions.jsonl"]),
                "--evidence-out",
                str(outputs["evidence.jsonl"]),
                "--routes-out",
                str(outputs["routes.jsonl"]),
                "--trace-out",
                str(outputs["trace.json"]),
            ]
            completed = subprocess.run(command, cwd=repo, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue(all(path.exists() for path in outputs.values()))

            prediction_rows = [json.loads(line) for line in outputs["predictions.jsonl"].read_text(encoding="utf-8").splitlines()]
            route_rows = [json.loads(line) for line in outputs["routes.jsonl"].read_text(encoding="utf-8").splitlines()]
            evidence_rows = [json.loads(line) for line in outputs["evidence.jsonl"].read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["id"] for row in prediction_rows}, {"p1", "n1"})
            self.assertEqual({row["id"] for row in route_rows}, {"p1", "n1"})
            self.assertEqual({row["id"] for row in evidence_rows}, {"p1", "n1"})
            self.assertEqual(next(row for row in route_rows if row["id"] == "n1")["action"], "raw")
            self.assertEqual(json.loads(outputs["trace.json"].read_text(encoding="utf-8"))["measurement_mode"], "replay")
            for path in outputs.values():
                self.assertNotIn("label", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
