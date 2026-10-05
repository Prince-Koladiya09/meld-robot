#!/usr/bin/env bash
# Downloads and unpacks MELD (raw videos + CSVs) into data/MELD.Raw
# MELD is released by the authors for research use: https://github.com/declare-lab/MELD
# If the URL below is dead, download manually and unpack so that data/MELD.Raw contains
# train_sent_emo.csv / dev_sent_emo.csv / test_sent_emo.csv and the *_splits*/ video folders.
set -euo pipefail

ROOT="${1:-data}"
URL="${MELD_URL:-http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz}"
mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -f MELD.Raw.tar.gz ] && [ ! -d MELD.Raw ]; then
  echo "Downloading MELD (~10 GB) from $URL"
  wget -c "$URL" -O MELD.Raw.tar.gz
fi

if [ ! -d MELD.Raw ]; then
  echo "Extracting outer archive"
  tar -xzf MELD.Raw.tar.gz
fi

cd MELD.Raw
for s in train dev test; do
  if [ -f "$s.tar.gz" ]; then
    echo "Extracting $s.tar.gz"
    tar -xzf "$s.tar.gz"
  fi
done
echo "Done. MELD is in $(pwd)"
