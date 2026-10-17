#!/usr/bin/env bash
set -eu

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BENCHMARKS=${*:-diabetes cancer housing silk-wave square-wave}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

mkdir "$WORK/original"
git -C "$REPO" archive f19cc63 | tar -x -C "$WORK/original"
make -C "$WORK/original" -j >/dev/null 2>&1
make -C "$REPO" -j >/dev/null 2>&1

for b in $BENCHMARKS; do
  (cd "$WORK/original" && ./genetic_benchmark "$b" >"$WORK/$b.original" 2>/dev/null) &
done
wait
for b in $BENCHMARKS; do
  (cd "$REPO" && ./genetic_benchmark "$b" >"$WORK/$b.optimized" 2>/dev/null)
done

status=0
for b in $BENCHMARKS; do
  t_orig=$(grep -o 'Time(.*' "$WORK/$b.original")
  t_opt=$(grep -o 'Time(.*' "$WORK/$b.optimized")
  if diff <(grep -v 'Time(' "$WORK/$b.original") \
          <(grep -v 'Time(' "$WORK/$b.optimized") >/dev/null 2>&1; then
    echo "SAME $b"
  else
    echo "DIFF $b"
    status=1
  fi
  echo "     original:  $t_orig"
  echo "     optimized: $t_opt"
done
exit $status
