# Implementation-Based Incentive Design for Autonomous Mobility-on-Demand and Transit Systems

Code for the paper *Implementation-Based Incentive Design for Autonomous Mobility-on-Demand and Transit Systems* ([arXiv:2605.18687](https://arxiv.org/abs/2605.18687)).

Given a socially optimal operating point of a multimodal network (AMoD + public transit + endogenous mode choice + road congestion),
the code computes the minimum payment that makes the point implementable. It does so by solving three oracles:
the social target (SO), the AMoD best response (SCA algorithm) and the transit best response (MIP).

<!-- TODO: one-line summary of the NYC/Manhattan case study + a figure -->

## Repository layout

```
main.py                 full pipeline: SO -> AMoD best response -> transit best response -> payment (k-implementation)
baseline_amod_br.py     AMoD best response against the historical (GTFS) transit frequencies
pricing.py              recovers the AMoD price from a solution
src/
  data_loader.py        loads the processed inputs
  social_optimum.py     social target oracle
  amod_br.py            AMoD best response (SCA)
  transit_br.py         transit best response
  transit_paths.py      candidate transit-path helpers (multi-path extension)
  constants.py          physical / economic constants
data_nyc/processed/     processed inputs read by the code (see "Data")
data_prep/              processing workflow for generating new datasets from raw sources (see data_prep/README.md)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Gurobi is required (`gurobipy`). The pip-installed restricted license is too small for the full instance; use an
[academic or full license](https://www.gurobi.com/academia/academic-program-and-licenses/).

## Data

### Raw sources
| Data | Source | Used for |
|---|---|---|
| TLC High-Volume FHV trip records, May 2024 (`fhvhv_tripdata_2024-05.parquet`) and taxi zone shapefile | <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page> | OD demand, zone geometry |
| NYC Transit subway GTFS | <https://www.mta.info/developers> | transit lines, frequencies, r5py routing |
| OpenStreetMap, Geofabrik New York extract (2026-03-01), clipped to Manhattan | <https://download.geofabrik.de/north-america/us/new-york.html> | walking network for r5py transit itineraries |
| OpenStreetMap, driving network via OSMnx (`graph_from_place`) | <https://www.openstreetmap.org> | AMoD road network |

`data_prep/download_raw.sh` downloads the raw files. Some of them change over time (the MTA feed, OSM); see [`data_prep/README.md`](data_prep/README.md) for the versions used and rebuilding limitations.

### Processed inputs (included in this repository)
`main.py` reads these from `data_nyc/processed/` (`--input_dir`):

| File | Content |
|---|---|
| `demand.csv` | Manhattan zone-to-zone trips, 9-11 AM on 2024-05-15 (TLC) |
| `on_demand/map.csv` | simplified driving network (edges with length, lanes, speed) |
| `on_demand/paths.csv` | candidate AMoD routes per OD (`path_id` suffix = rank) |
| `metro/optimization_inputs.pkl.gz` | one transit path per OD + line topology (`--num_path 1`) |
| `metro/optimization_inputs_k5.pkl.gz` | up to 5 candidate transit paths per OD (`--num_path > 1`) |
| `metro/gtfs_frequencies.pkl` | historical line frequencies (used by `baseline_amod_br.py`) |

**Use the supplied processed files to reproduce the paper's experimental inputs.**
The [`data_prep/`](data_prep/README.md) scripts document the processing workflow and support generating
alternative datasets in the format expected by the experiment code. They are not an exact reconstruction
of the supplied files: source-data updates, routing and graph-processing variability, and differences in
reference-path selection can change OD pairs, candidate routes, and network attributes.

When generating a new dataset, rebuild all dependent files together in a separate output directory.
Do not mix rebuilt transit files with supplied road files, since their OD sets may differ. Point the
experiment code to the resulting directory with `--input_dir`. See the data-preparation README for
commands, compatible-input requirements, and validation limits.

## Usage

```bash
# full pipeline at background traffic 0.5, AMoD cost 0.6, entropy regularization theta = 0.5
python main.py --stage all --bl 0.5 --ca 0.6 --cl 320 --theta 0.5 -o results/example

# two-stage: solve and save the social optimum, then run both best responses
python main.py --stage so -o results/example ...
python main.py --stage br -o results/example ...

# multi-path extension: up to 3 candidate transit paths per OD
python main.py --num_path 3 --transit_file metro/optimization_inputs_k5.pkl.gz ...

# AMoD best response against the historical transit schedule
python baseline_amod_br.py --bl 0.5 --ca 0.6 ...
```

`python main.py --help` lists all arguments (BPR parameters `--alpha/--beta`, default 0.8 / 3, regularization `--reg/--theta`, `--mip` gap, `--threads`, `--time_limit`, ...).

### Output
Files are written to `--output_dir`, suffixed with `_{bl}_{ca}_{cl}[_ent_theta{theta}]`:

| File | Content |
|---|---|
| `so_summary*.csv`, `mode_split*.csv` | social target: aggregate cost and per-OD mode split |
| `amod_br_mode_split*.csv`, `transit_br_mode_split*.csv` | best-response solutions per OD |
| `k_summary*.csv` | payment decomposition (AMoD and transit deviation gains) |
| `sca_convergence*.pkl` | SCA convergence history |

## Reproducing the paper

Use the supplied `data_nyc/processed/` files; rerunning data preparation generates a potentially different
experimental instance and may therefore produce different results.

<!-- TODO: map figures/tables to the commands; link to the results archive (Zenodo) -->

## Citation

```bibtex
@article{li2026implementation,
  title={Implementation-Based Incentive Design for Autonomous Mobility-on-Demand and Transit Systems},
  author={Li, Xinling and Zhang, Runyu and Zardini, Gioele},
  journal={arXiv preprint arXiv:2605.18687},
  year={2026}
}
```

## License

<!-- TODO: code license. Note: on_demand/map.csv and paths.csv derive from OpenStreetMap (ODbL, (c) OpenStreetMap contributors). -->
