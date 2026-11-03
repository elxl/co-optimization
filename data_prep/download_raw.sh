#!/usr/bin/env bash
# Download the raw NYC inputs into data_nyc/ (or the directory given as $1). Existing files are never overwritten.
#   TLC FHV trips (May 2024) + taxi zones : https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
#   MTA subway GTFS                       : https://www.mta.info/developers
#   OpenStreetMap, Geofabrik New York     : https://download.geofabrik.de/north-america/us/new-york.html
# The MTA feed and the Geofabrik extract are moving targets (see data_prep/README.md): the exact files used
# in the paper are in the data archive, which is the way to reproduce the processed inputs bit for bit.
set -euo pipefail

RAW=${1:-data_nyc}
GTFS=f-dr5r-nyctsubway-latest
mkdir -p "$RAW/demand"


fetch() {  # fetch <url> <destination>
    if [ -e "$2" ]; then echo "exists, skipping: $2"; else echo "downloading: $2"; curl -fL --retry 3 -o "$2" "$1"; fi
}

fetch https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-05.parquet "$RAW/demand/fhvhv_tripdata_2024-05.parquet"

if [ ! -e "$RAW/taxi_zones/taxi_zones.shp" ]; then
    fetch https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip "$RAW/taxi_zones.zip"
    unzip -q "$RAW/taxi_zones.zip" -d "$RAW" && rm "$RAW/taxi_zones.zip"
fi

fetch https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip "$RAW/$GTFS.zip"  # r5py reads the zip
[ -e "$RAW/$GTFS/stop_times.txt" ] || unzip -q "$RAW/$GTFS.zip" -d "$RAW/$GTFS"

fetch https://download.geofabrik.de/north-america/us/new-york-latest.osm.pbf "$RAW/new-york-latest.osm.pbf"
