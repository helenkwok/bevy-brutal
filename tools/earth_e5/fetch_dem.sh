#!/bin/bash
# Earth round 5: Copernicus GLO-30 tiles N44-N47 x E005-E009 (20 tiles, ~930 MB) into
# /workspace/data/earth/dem, then mosaic (run dem_mosaic.py there). AWS open data, no auth.
set -e
D=${1:-/workspace/data/earth/dem}; mkdir -p "$D"; cd "$D"
for la in 44 45 46 47; do for lo in 005 006 007 008 009; do
  n=Copernicus_DSM_COG_10_N${la}_00_E${lo}_00_DEM
  [ -s $n.tif ] || curl -sf --retry 3 -o $n.tif https://copernicus-dem-30m.s3.amazonaws.com/$n/$n.tif
done; done
echo "tiles in $D; next: (cd $D && python3 dem_mosaic.py)"
