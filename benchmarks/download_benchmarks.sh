#!/usr/bin/env bash
# Downloads the public benchmark files used in Section 5.14.
set -e
cd "$(dirname "$0")"
BASE=https://raw.githubusercontent.com/ChuanbinZhang/Multi-view-datasets/master
for f in 3Sources BBCSport MSRC-v5 Handwritten Caltech101-7; do
  curl -L -o "$f.mat" "$BASE/$f.mat"
done
curl -L -o 20newsgroups.mat https://raw.githubusercontent.com/zskong/Multi-view-data/main/data/20newsgroups.mat   # NGs (R3-2)
curl -L -o penguins.csv https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv
sha256sum -c SHA256SUMS.txt