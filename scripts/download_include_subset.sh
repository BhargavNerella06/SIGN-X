#!/usr/bin/env bash
# Resumable, checksum-verified download of the INCLUDE subset used by
# backend/experiments/isl_include_subset28.yaml (Greetings, Pronouns, Colours).
#
# Safe to re-run at any time: partial files are resumed with HTTP range
# requests (curl -C -), and a file is only considered done once its MD5
# matches the checksum published in the Zenodo record 4010759 API.
#
# Usage (from repo root, Git Bash):  bash scripts/download_include_subset.sh [PARALLEL]
set -u
PARALLEL="${1:-3}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/backend/datasets/_include_zips"
mkdir -p "$DEST"
cd "$DEST" || exit 1

# name  size  md5  (from https://zenodo.org/api/records/4010759)
MANIFEST="Greetings_1of2.zip 1573733886 457624ce89b2d32520d6b61bd1f113a2
Greetings_2of2.zip 1208700458 2dd843662545b0ca57e79cff84562698
Pronouns_1of2.zip 1419853451 854b5510d0fd07b8c85cb4dda37492b6
Pronouns_2of2.zip 946837765 5fdb1cc5266be67136aca15509c978ba
Colours_1of2.zip 1268663334 097cf76e442f92a66ff71cad5738b409
Colours_2of2.zip 1452446819 1ac3c8d153c0b77156dfcca9cc72747e"

fetch() {
  local name="$1" size="$2" md5="$3"
  local url="https://zenodo.org/api/records/4010759/files/${name}/content"
  if [ -f "$name.ok" ]; then echo "[skip] $name already verified"; return 0; fi
  local attempt=0
  while [ "$attempt" -lt 50 ]; do
    attempt=$((attempt + 1))
    local have=0
    [ -f "$name" ] && have=$(stat -c %s "$name")
    if [ "$have" -lt "$size" ]; then
      echo "[$(date '+%F %T')] $name: resuming at $have/$size (attempt $attempt)"
      curl -sSL --fail -C - --retry 5 --retry-delay 10 --speed-limit 10000 --speed-time 120 \
        -o "$name" "$url" || { sleep 15; continue; }
      have=$(stat -c %s "$name")
    fi
    if [ "$have" -gt "$size" ]; then
      echo "[$(date '+%F %T')] $name: larger than expected ($have > $size), restarting"; rm -f "$name"; continue
    fi
    if [ "$have" -eq "$size" ]; then
      local got; got=$(md5sum "$name" | cut -d' ' -f1)
      if [ "$got" = "$md5" ]; then
        touch "$name.ok"; echo "[$(date '+%F %T')] $name: DONE, md5 verified"; return 0
      fi
      echo "[$(date '+%F %T')] $name: md5 mismatch ($got != $md5), restarting"; rm -f "$name"
    fi
  done
  echo "[$(date '+%F %T')] $name: FAILED after $attempt attempts"; return 1
}
export -f fetch

echo "$MANIFEST" | xargs -P "$PARALLEL" -L 1 bash -c 'fetch "$0" "$1" "$2"'

echo "=== summary ==="
fail=0
while read -r name size md5; do
  if [ -f "$name.ok" ]; then echo "OK      $name"; else echo "MISSING $name"; fail=1; fi
done <<< "$MANIFEST"
exit $fail
