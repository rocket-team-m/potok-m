#!/usr/bin/env bash
# Правило репозитория: бинарники и тяжёлые файлы живут в Releases, а не в git.
set -euo pipefail

fail=0
forbidden='\.(sldprt|sldasm|slddrw|cas|dat|trn|msh|wbpj|bag|mcap|zip|7z|tar|gz)$'

while IFS= read -r f; do
  echo "::error file=$f::Бинарный артефакт в git. Выложи его в Releases."
  fail=1
done < <(git ls-files | grep -Ei "$forbidden" || true)

while IFS= read -r f; do
  [ -f "$f" ] || continue
  size=$(stat -c%s "$f")
  if [ "$size" -gt 2097152 ]; then
    echo "::error file=$f::Файл больше 2 МБ. Выложи его в Releases."
    fail=1
  fi
done < <(git ls-files)

if [ "$fail" -eq 0 ]; then
  echo "OK: тяжёлых файлов в git нет."
fi
exit "$fail"
