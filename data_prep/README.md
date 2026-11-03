# Data preparation

**Use the supplied `data_nyc/processed/` files to reproduce the paper's experimental inputs.**
This folder documents the processing workflow and supports generating alternative datasets from raw sources
in the format expected by the experiment code. Running these scripts is not guaranteed to reproduce the
supplied files exactly; see [Reproducibility](#reproducibility) below.

To generate a new dataset, run from the repository root and write all dependent outputs to the same
separate directory. Do not mix rebuilt transit files with supplied road files: the OD sets can differ.
Raw-data download links are in `download_raw.sh`. When using a newly downloaded GTFS feed, choose a
`--departure` date within its service calendar for step 03; the default `2026-05-15T09:00` applies to the
feed used in the paper.

```bash
pip install -r data_prep/requirements.txt         # + Java 21 for r5py, osmium-tool for step 02
bash   data_prep/download_raw.sh                  # raw data into data_nyc/ (skips files that exist)

PREP_OUT=data_nyc/processed_rebuilt
python data_prep/01_demand.py --out_dir "$PREP_OUT"
# Skip step 02 if data_nyc/manhattan.osm.pbf already exists.
bash   data_prep/02_clip_osm.sh
python data_prep/03_transit_itineraries.py --out_dir "$PREP_OUT"  # add --departure for a different GTFS feed
python data_prep/04_transit_inputs.py --out_dir "$PREP_OUT"
python data_prep/05_gtfs_frequencies.py --out_dir "$PREP_OUT"
python data_prep/06_amod_network.py --out_dir "$PREP_OUT"

# Run an experiment with the newly generated collection of inputs:
python main.py --stage all --input_dir "$PREP_OUT" -o results/rebuilt_example
```

The Python scripts take `--raw_dir` (default `data_nyc`) and `--out_dir` (default `data_nyc/processed`)
and **overwrite** their outputs. The commands above keep the supplied processed files intact.
`python data_prep/<script> --help` lists the remaining options. The shell scripts use positional arguments:
`download_raw.sh [raw_dir]` and `02_clip_osm.sh [input.osm.pbf] [output.osm.pbf]`; the clipping command
refuses to overwrite an existing output. Step 06's `--graphml` path is separate from `--raw_dir` and
defaults to `data_nyc/manhattan_drive.graphml`.

| Step | Reads | Writes | What it does |
|---|---|---|---|
| `01_demand.py` | TLC FHV parquet, taxi zones | `demand.csv` | trips 09:00-11:00 on 2024-05-15 between the 64 modelled Manhattan zones, self-pairs dropped, counted per OD |
| `02_clip_osm.sh` | Geofabrik New York extract | `manhattan.osm.pbf` | bounding-box clip around Manhattan (walking network for r5py) |
| `03_transit_itineraries.py` | `demand.csv`, `manhattan.osm.pbf`, GTFS zip, taxi zones | `metro/itineraries_raw_multi.csv.gz` | r5py transit + walk itineraries (all options) between zone centroids, for every OD |
| `04_transit_inputs.py` | raw itineraries, `demand.csv`, GTFS | `metro/optimization_inputs.pkl.gz`, `metro/optimization_inputs_k5.pkl.gz` | ranks distinct subway paths per OD, filters, extracts lines and directed segments used; line topology from GTFS |
| `05_gtfs_frequencies.py` | GTFS, `optimization_inputs.pkl.gz` | `metro/gtfs_frequencies.{csv,pkl}` | AM-peak trips per line, snapped to the frequency set (baseline without subsidy) |
| `06_amod_network.py` | taxi zones, `optimization_inputs.pkl.gz` | `on_demand/map.csv`, `on_demand/paths.csv` | OSMnx driving network, 5 penalty-diversified shortest routes per OD |

Dependencies: 01 and 02 are independent; 03 needs 01 and 02; 04 needs 03; 05 and 06 need 04.
Choices that define the dataset (excluded zones, filters, window, departure time) are command-line options or constants
at the top of `common.py`, `04_transit_inputs.py`:

- **Zones.** Manhattan zones except Randalls Island (194), Roosevelt Island (202) and the Governor's/Ellis/Liberty Island zones (103-105).
- **Path filters** (04). At most 1200 m walking, 1-3 subway legs, ODs with at least 4 trips. An OD is kept if its best path passes. The top `--k` distinct paths are selected before filtering, so fewer than k candidates may remain.
- **Line topology** (04). The single-path file keeps one representative trip per line (original behaviour); the k-path file uses the union of all trip patterns. The sweeps use the k-path file for every K, so the line topology is identical across them.
- **Driving network** (06). Roads of type motorway, trunk, primary, secondary, tertiary and unclassified; intersections consolidated within 5 m; largest strongly connected component. Edges without an OSM tag get 2 lanes / 40 mph (about 18% / 35% of the edges); the speed is set by `--default_speed` (mph).

## Reproducibility

The supplied processed files are the reference for the paper. A successful run with compatible raw inputs
is intended to generate a new collection of experiment inputs, not necessarily the same experimental
instance. Differences can affect demand coverage, candidate paths, capacities, travel times, and results.
Exact reconstruction would require the historical raw inputs, preserved intermediate paths, original
road graph and zone-to-node mapping, and a matching processing environment.
<!-- TODO: publish and link a versioned archive of historical inputs and intermediate artifacts. -->

- **Raw data that moves.**
  - *GTFS.* The MTA replaces its feed continuously. The feed used here was valid 2025-12-08 to 2026-05-16, which is why r5py departs on 2026-05-15 (a Friday) while the demand is from 2024-05-15. A feed downloaded later has a different validity window; pass a `--departure` inside it to `03_transit_itineraries.py`.
  - *Geofabrik.* The dated extract used (`new-york-260301`) is no longer served; `download_raw.sh` fetches `new-york-latest`.
  - *OpenStreetMap driving network.* `06` queries OSM live and saves the raw graph to `--graphml`; later runs load it.
- **The Manhattan clip is approximate.** `manhattan.osm.pbf` was clipped with osmium, but the command was not recorded. `02_clip_osm.sh` uses a bounding box measured from that file; a re-clip is slightly larger (1.58M nodes vs 1.56M).
- **Routing and reference-path selection.** Routing runs can differ in walking distances and selected paths, moving ODs across the 1200 m filter. The notebook preserved the original single-path results when constructing its multi-path dataset; step 04 currently selects all paths from the provided raw itinerary table. Even with the stored raw itineraries, this changes the reference OD set and candidate paths.
- **Road processing.** Merged OSM lane/speed tags can have an unordered representation, and step 06 takes the first value. Nearest-node assignments can also change when two nodes are nearly equidistant. Saving a newly generated GraphML makes later runs more consistent but does not recover the historical choices.
