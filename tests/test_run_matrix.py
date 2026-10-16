import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


RUNNER = Path(__file__).resolve().parents[1] / "benchmark/run_matrix.py"


class MatrixRecovery(unittest.TestCase):
    def test_partial_resume_aliases_and_changed_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("src", "include", "benchmark"):
                (root / directory).mkdir()
            source = root / "src/dummy.cpp"
            source.write_text("// fixture\n")
            (root / "benchmark/genetic_benchmark.cpp").write_text("// fixture\n")
            compiler = root / "compiler"
            compiler.write_text('''#!/usr/bin/env python3
import os, pathlib, sys
if '--version' in sys.argv:
    print('fixture compiler 1')
else:
    with open(os.environ['BUILD_COUNT'], 'a') as f: f.write('build\\n')
    p = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])
    p.write_text("#!/usr/bin/env python3\\nimport os\\nwith open(os.environ['RUN_COUNT'], 'a') as f: f.write('run\\\\n')\\nprint('#lang racket')\\nprint(';;Time = ignored')\\n")
    p.chmod(0o755)
''')
            compiler.chmod(0o755)
            environment = dict(os.environ, GCC_CXX=str(compiler),
                               BUILD_COUNT=str(root / "builds"),
                               RUN_COUNT=str(root / "runs"))
            output = root / "results"
            command = ["python3", str(RUNNER), "--source-root", str(root),
                       "--output", str(output), "--compilers", "gcc",
                       "--workloads", "diabetes", "--warmups", "1", "--repeats", "2"]

            def run(*arguments):
                return subprocess.run(command + list(arguments), env=environment,
                                      capture_output=True, text=True)

            initial = run("--levels", "0")
            self.assertEqual(initial.returncode, 0, initial.stderr)
            path = output / "results.json"
            report = json.loads(path.read_text())
            report["runs"].pop()
            report["summaries"].clear()
            path.write_text(json.dumps(report))
            resumed = run("--levels", "0", "1", "2", "--resume", "--skip-identical")
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            report = json.loads(path.read_text())
            self.assertEqual(len(report["runs"]), 3)
            self.assertEqual(len(report["summaries"]), 1)
            self.assertEqual((root / "runs").read_text().splitlines(), ["run"] * 4)
            self.assertEqual((root / "builds").read_text().splitlines(), ["build"] * 3)
            for build in report["builds"][1:]:
                self.assertEqual(build["timing_alias_of"],
                                 {"compiler": "gcc", "requested_level": 0})
            rejected = run("--levels", "0", "--resume", "--extra-flags=-g")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("changed or failed build", rejected.stderr)
            source.write_text("// modified\n")
            rejected = run("--levels", "0", "--resume")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("source_sha256 changed", rejected.stderr)
            source.write_text("// fixture\n")
            (output / report["runs"][0]["stdout"]).write_text("altered\n")
            rejected = run("--levels", "0", "--resume")
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("altered output", rejected.stderr)
            self.assertEqual((root / "runs").read_text().splitlines(), ["run"] * 4)


if __name__ == "__main__":
    unittest.main()
