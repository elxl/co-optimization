# Implementation-Based Incentive Design for Autonomous Mobility-on-Demand and Transit Systems

This repository includes code for the paper *Implementation-Based Incentive Design for Autonomous Mobility-on-Demand and Transit Systems* ([arXiv:2605.18687](https://arxiv.org/abs/2605.18687)).

Given a socially optimal operating point of a multimodal network, the code computes the minimum payment that makes the point implementable. It does so by solving three oracles: the social target, the AMoD best response, and the transit best response.


## Repository layout

```
main.py                 full pipeline: SO -> AMoD best response -> transit best response -> payment (k-implementation)
baseline_amod_br.py     AMoD best response against the historical (GTFS) transit frequencies
pricing.py              the price recovery step of AMoD best response
src/
  data_loader.py        loads the processed inputs
  social_optimum.py     social target oracle
  amod_br.py            AMoD best response
  transit_br.py         transit best response
  transit_paths.py      candidate transit-path helpers (multi-path extension)
  constants.py          physical / economic constants
data_nyc/processed/     processed inputs read by the code (see "Data")
data_prep/              processing workflow for generating datasets into expected formats from raw sources (see data_prep/README.md)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Gurobi is required (`gurobipy`). You will need an
[academic or full license](https://www.gurobi.com/academia/academic-program-and-licenses/) to solve large cases (e.g., Manhattan).

## Data

### Raw sources
| Data | Source | Used for |
|---|---|---|
| TLC High-Volume FHV trip records, May 2024 (`fhvhv_tripdata_2024-05.parquet`) and taxi zone shapefile | <https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page> | OD demand, zone geometry |
| NYC Transit subway GTFS | <https://www.mta.info/developers> | transit lines, frequencies, r5py routing |
| OpenStreetMap, Geofabrik New York extract (2026-03-01), clipped to Manhattan | <https://download.geofabrik.de/north-america/us/new-york.html> | walking network for r5py transit itineraries |
| OpenStreetMap, driving network via OSMnx (`graph_from_place`) | <https://www.openstreetmap.org> | AMoD road network |

`data_prep/download_raw.sh` downloads the raw files. Some of them change over time (the MTA feed, OSM); see [`data_prep/README.md`](data_prep/README.md) for the versions used and rebuilding limitations.

### Processed inputs
`main.py` reads these from `data_nyc/processed/` (`--input_dir`):

| File | Content |
|---|---|
| `demand.csv` | Manhattan zone-to-zone trips, 9-11 AM on 2024-05-15 (TLC) |
| `on_demand/map.csv` | simplified driving network (edges with length, lanes, speed) |
| `on_demand/paths.csv` | candidate AMoD routes per OD (`path_id` suffix = rank) |
| `metro/optimization_inputs.pkl.gz` | one transit path per OD + line topology (`--num_path 1`) |
| `metro/optimization_inputs_k5.pkl.gz` | up to 5 candidate transit paths per OD (`--num_path > 1`) |
| `metro/gtfs_frequencies.pkl` | historical line frequencies (used by `baseline_amod_br.py`) |

**Use the supplied processed files to reproduce the paper's experiments.**
The [`data_prep/`](data_prep/README.md) scripts document the processing workflow and support generating
alternative datasets in the format expected by the experiment code. They are not an exact reconstruction
of the supplied processed files under `data/processed`. Source-data updates, routing and graph-processing variability, and differences in
reference-path selection can change OD pairs, candidate routes, and network attributes.

When generating a new dataset, rebuild all dependent files together in a separate output directory.
Do not mix rebuilt transit files with supplied road files, since their OD sets may differ. Point the
experiment code to the resulting directory with `--input_dir`. See the data-preparation README for
commands, compatible-input requirements, and validation limits.

## Usage

`python main.py --help` lists all arguments. The core inputs used in the experiment scripts under `scripts/` are:

- **`--bl`**: Background road traffic as a fraction of each edge's capacity (`b_e = bl × capacity_e`).
- **`--ca`**: AMoD operating cost per vehicle-kilometre. 
- **`--cl`**: Transit operating-cost coefficient; a line's hourly operating cost is `cl × frequency`. 
- **`--pt_cap`**: Transit capacity in passengers per departure.
- **`--alpha`, `--beta`**: BPR road-congestion parameters in `travel_time = free_flow_time × (1 + alpha × (flow / capacity)^beta)`. 
- **`--reg`, `--theta`**: Social-target regularization type and scale. `--reg 0` disables regularization, `1` selects quadratic regularization, and `2` selects entropy regularization. The regularization term is weighted by `1 / theta`; theta is ignored when `reg=0`. The paper uses `--reg 2`.
- **`--num_path`**: Maximum number of candidate transit paths per OD. The paper uses 5; an OD can have fewer available paths.
- **`--transit_file`**: Transit input file, relative to `--input_dir`. The command-line default is another separate single-path file, `metro/optimization_inputs.pkl.gz`.
- **`--num_path_amod`**: Optional cap on candidate AMoD routes per OD. By default, all available routes are retained.
- **`--input_dir`**: Directory containing the processed input collection; defaults to `data_nyc/processed`.
- **`--output_dir` / `-o`**: Directory for saved solutions and results. The filename suffix is `_{bl}_{ca}_{cl}` for `--reg 0`, `_{bl}_{ca}_{cl}_quad_theta{theta}` for `--reg 1`, and `_{bl}_{ca}_{cl}_ent_theta{theta}` for `--reg 2`. For experiments that vary transit capacity, path counts, or other settings absent from this suffix, use separate output directories or modify the `suffix` construction in `main.py` to include those settings and avoid overwriting results. 
- **`--threads`, `--time_limit`**: Gurobi thread count and time limit in seconds.
- **`--stage`**: `all` (default) solves the social optimum, computes both operators' best responses, and evaluates the payment; `so` solves and saves only the social optimum; `br` loads an existing social-optimum solution and computes both best responses and the payment.
- **`--load_so`**: Explicit path to a saved social-optimum solution, used to reuse a target from an earlier run.

**Before running `--stage br`, the saved social-optimum result must exist.** Each operator's best response
holds the other operator's strategy at its social-optimum target. Both operators' target strategies and
states are stored together in `so_solution<suffix>.pkl`, produced by `--stage so` or `--stage all`.
The `br` stage reads this file from `--output_dir`, or from the explicit path supplied with `--load_so`;
separate best-response result files are not required. Use the same input data and model settings as the
saved target. For example, with `--bl 0.5 --ca 0.6 --cl 320 --reg 2 --theta 0.5`, the expected filename is
`so_solution_0.5_0.6_320.0_ent_theta0.5.pkl`.

```bash
# full pipeline at background traffic 0.5, AMoD cost 0.6, entropy regularization theta = 0.5
python main.py --stage all --bl 0.5 --ca 0.6 --cl 320 --theta 0.5 -o results/example

# two-stage: solve and save the social optimum, then run both best responses
python main.py --stage so --bl 0.5 --ca 0.6 --cl 320 --reg 2 --theta 0.5 -o results/example
python main.py --stage br --bl 0.5 --ca 0.6 --cl 320 --reg 2 --theta 0.5 -o results/example

# multi-path extension: up to 3 candidate transit paths per OD
python main.py --num_path 3 --transit_file metro/optimization_inputs_k5.pkl.gz ...

# AMoD best response against the historical transit schedule
python baseline_amod_br.py --bl 0.5 --ca 0.6 ...
```

### Output
Files are written to `--output_dir` using the suffix described above, inserted before the file extension:

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
