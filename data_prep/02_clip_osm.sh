#!/usr/bin/env bash
# Clip the Geofabrik New York extract to a Manhattan bounding box (needs `osmium-tool`).
# The bounding box was measured from the clipped file used in the paper, not recorded when it was
# made, so a re-clip is slightly larger (see data_prep/README.md). The original manhattan.osm.pbf is
# part of the data archive. osmium refuses to overwrite an existing output file.
# Usage: bash data_prep/02_clip_osm.sh [input.osm.pbf] [output.osm.pbf]
set -euo pipefail

IN=${1:-data_nyc/new-york-latest.osm.pbf}
OUT=${2:-data_nyc/manhattan.osm.pbf}

osmium extract -b -74.027,40.682,-73.906,40.882 "$IN" -o "$OUT"
