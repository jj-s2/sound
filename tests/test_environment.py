import unittest

from xh202615.environment import collect_environment


class EnvironmentTest(unittest.TestCase):
    def test_report_contains_required_sections_and_explicit_missing_package(self):
        report = collect_environment(
            ["definitely-not-installed-xh202615", "json"],
            ["missing-artifact-xh202615.bin"],
        )

        for key in (
            "python_version",
            "platform",
            "executable",
            "packages",
            "torch",
            "cuda",
            "device",
            "resource_capabilities",
            "artifact_checks",
        ):
            self.assertIn(key, report)

        self.assertEqual(
            report["packages"]["definitely-not-installed-xh202615"],
            {"installed": False, "version": None},
        )
        self.assertFalse(report["artifact_checks"]["missing-artifact-xh202615.bin"]["exists"])

    def test_package_version_resolver_can_be_injected(self):
        def resolver(name):
            if name == "present":
                return "1.2.3"
            raise RuntimeError("not found")

        report = collect_environment(["present", "absent"], [], version_resolver=resolver)
        self.assertEqual(report["packages"]["present"], {"installed": True, "version": "1.2.3"})
        self.assertEqual(report["packages"]["absent"], {"installed": False, "version": None})


if __name__ == "__main__":
    unittest.main()
