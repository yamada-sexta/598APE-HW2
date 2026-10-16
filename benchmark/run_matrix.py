#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import statistics
import subprocess
import time

WORKLOADS = ["diabetes", "cancer", "housing", "silk-wave", "square-wave"]


def normalized_hash(path):
    lines = path.read_bytes().splitlines(keepends=True)
    return hashlib.sha256(b"".join(line for line in lines
                                  if not line.startswith(b";;Time"))).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Build and time the original workloads; never run timed jobs concurrently.")
    parser.add_argument("--source-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compilers", nargs="+", choices=["gcc", "clang", "opencilk"],
                        default=["gcc", "clang"])
    parser.add_argument("--levels", nargs="+", type=int, choices=range(6), default=list(range(6)))
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=WORKLOADS)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Per-execution limit; timeouts are failures, not measurements")
    parser.add_argument("--reference-results", type=Path)
    parser.add_argument("--workers", nargs="+", type=int, default=[1])
    parser.add_argument("--extra-flags", default="",
                        help="Additional compile/link flags; argv parsed without a shell")
    parser.add_argument("--resume", action="store_true",
                        help="Reuse successful runs after checking source, flags, and output")
    parser.add_argument("--skip-identical", action="store_true",
                        help="Build each requested level but time identical binaries only once")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or any(w < 1 for w in args.workers):
        parser.error("Require positive repeats/workers and nonnegative warmups")
    source = args.source_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    references = {}
    if args.reference_results:
        prior = json.loads(args.reference_results.read_text())
        for row in prior["runs"]:
            if row.get("exit") == 0:
                references.setdefault(row["workload"], row["normalized_sha256"])
    report = {"source_root": str(source), "source_commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source, capture_output=True, text=True).stdout.strip(),
        "source_diff_sha256": hashlib.sha256(subprocess.run(
            ["git", "diff", "HEAD"], cwd=source, capture_output=True).stdout).hexdigest(),
        "machine": dict(zip(("sysname", "nodename", "release", "version", "machine"), os.uname())), "warmups": args.warmups,
        "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(list((source / "src").glob("*")) +
                                          list((source / "include").glob("*.h")) +
                                          [source / "benchmark/genetic_benchmark.cpp"])
                          if p.is_file()},
        "repeats": args.repeats, "builds": [], "runs": [], "summaries": []}
    result_file = output / "results.json"
    if args.resume:
        if not result_file.exists():
            parser.error("--resume requires an existing results.json")
        previous = json.loads(result_file.read_text())
        for key in ("source_root", "source_sha256", "machine", "warmups", "repeats"):
            if previous[key] != report[key]:
                parser.error(f"Cannot resume: {key} changed")
        for row in previous["runs"]:
            path = output / row["stdout"]
            if (row["exit"] != 0 or not row["matches_reference"] or not path.exists()
                    or normalized_hash(path) != row["normalized_sha256"]):
                parser.error("Cannot resume failed, missing, or altered output")
            reference = references.setdefault(row["workload"], row["normalized_sha256"])
            if reference != row["normalized_sha256"]:
                parser.error("Cannot resume: reference output changed")
        previous.setdefault("resumptions", []).append({
            "source_commit": report["source_commit"],
            "source_diff_sha256": report["source_diff_sha256"],
            "skip_identical": args.skip_identical})
        report = previous
    elif result_file.exists():
        parser.error("Output already contains results; use --resume or a new directory")

    def save():
        temporary = output / "results.json.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output / "results.json")

    for compiler in args.compilers:
        executable = {"gcc": os.environ.get("GCC_CXX", "g++"),
                      "clang": os.environ.get("CLANG_CXX", "clang++"),
                      "opencilk": os.environ.get("OPENCILK_CXX", "/opt/opencilk/bin/clang++")}[compiler]
        if not shutil.which(executable):
            raise SystemExit(f"Compiler unavailable: {executable}")
        version = subprocess.check_output([executable, "--version"], text=True)
        for level in args.levels:
            name = f"{compiler}-O{level}"
            binary = output / name
            command = [executable, "-std=c++17", *shlex.split(args.extra_flags),
                       f"-O{level}", "-ffp-contract=off",
                       "-Wall", "-Wextra", "-Iinclude", "-I."]
            if compiler == "opencilk":
                command += ["-fopencilk", "-DGENETIC_USE_CILK"]
            command += [str(p.relative_to(source)) for p in sorted((source / "src").glob("*.cpp"))]
            command += ["benchmark/genetic_benchmark.cpp", "-o", str(binary)]
            build_record = next((b for b in report["builds"]
                                 if b["compiler"] == compiler and b["requested_level"] == level), None)
            if build_record is not None:
                if (build_record["command"] != command or build_record["version"] != version
                        or build_record["exit"] != 0 or not binary.exists()
                        or hashlib.sha256(binary.read_bytes()).hexdigest() != build_record["binary_sha256"]):
                    parser.error(f"Cannot resume changed or failed build: {name}")
            else:
                start = time.monotonic()
                build = subprocess.run(command, cwd=source, capture_output=True, text=True)
                (output / f"{name}-build.log").write_text(build.stdout + build.stderr)
                build_record = {"compiler": compiler, "requested_level": level,
                    "version": version, "command": command, "exit": build.returncode,
                    "seconds": time.monotonic() - start,
                    "diagnostics": build.stdout + build.stderr,
                    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()
                        if build.returncode == 0 else None}
                report["builds"].append(build_record)
            save()
            if build_record["exit"]:
                raise SystemExit(f"Build failed: {name}; see build log")
            equivalent = next((b for b in report["builds"]
                               if b["compiler"] == compiler and b["requested_level"] != level
                               and b["exit"] == 0 and not b.get("timing_alias_of")
                               and b["binary_sha256"] == build_record["binary_sha256"]
                               and b["requested_level"] in args.levels), None)
            if args.skip_identical and equivalent is not None:
                build_record["timing_alias_of"] = {
                    "compiler": compiler, "requested_level": equivalent["requested_level"]}
                save()
                print(f"{name}: identical binary; timings belong to {compiler}-O{equivalent['requested_level']}", flush=True)
                continue
            for workers in args.workers if compiler == "opencilk" else [1]:
                environment = dict(os.environ, CILK_NWORKERS=str(workers))
                for workload in args.workloads:
                    if any(s["compiler"] == compiler and s["requested_level"] == level
                           and s["workers"] == workers and s["workload"] == workload
                           for s in report["summaries"]):
                        continue
                    samples = []
                    for iteration in range(args.warmups + args.repeats):
                        prior_run = next((r for r in report["runs"]
                                          if r["compiler"] == compiler and r["requested_level"] == level
                                          and r["workers"] == workers and r["workload"] == workload
                                          and r["iteration"] == iteration), None)
                        if prior_run is not None:
                            if iteration >= args.warmups:
                                samples.append(prior_run)
                            continue
                        prefix = f"{name}-w{workers}-{workload}-{iteration}"
                        stdout = output / f"{prefix}.rkt"
                        stderr = output / f"{prefix}.stderr"
                        timing = output / f"{prefix}.time"
                        run_command = ["/usr/bin/time", "-f", "%e %M", "-o", str(timing),
                                       "timeout", str(args.timeout), str(binary), workload]
                        with stdout.open("wb") as out, stderr.open("wb") as err:
                            run = subprocess.run(run_command, cwd=source, env=environment,
                                                 stdout=out, stderr=err)
                        elapsed, rss = timing.read_text().splitlines()[-1].split()
                        digest = normalized_hash(stdout)
                        reference = references.setdefault(workload, digest) if run.returncode == 0 else None
                        row = {"compiler": compiler, "requested_level": level,
                               "workers": workers, "workload": workload,
                               "iteration": iteration, "warmup": iteration < args.warmups,
                               "command": run_command, "exit": run.returncode,
                               "wall_seconds": float(elapsed), "peak_rss_kib": int(rss),
                               "normalized_sha256": digest,
                               "matches_reference": digest == reference,
                               "stdout": stdout.name, "stderr": stderr.name}
                        report["runs"].append(row)
                        save()
                        print(f"{prefix}: {elapsed}s RSS={rss}KiB match={digest == reference}", flush=True)
                        if run.returncode or digest != reference:
                            raise SystemExit(f"Execution/reference failure: {prefix}; evidence preserved")
                        if iteration >= args.warmups:
                            samples.append(row)
                    summary = {"compiler": compiler, "requested_level": level,
                               "workers": workers, "workload": workload,
                               "median_seconds": statistics.median(r["wall_seconds"] for r in samples),
                               "min_seconds": min(r["wall_seconds"] for r in samples),
                               "max_seconds": max(r["wall_seconds"] for r in samples),
                               "peak_rss_kib": max(r["peak_rss_kib"] for r in samples)}
                    report["summaries"].append(summary)
                    save()
                    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
