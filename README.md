# GLM Functional Element Discovery

## Project purpose

This repository evaluates sequence language models on prepared ribosome
sequences and annotation intervals. It currently implements:

- masked single-base reconstruction;
- nucleotide dependency maps.

The implemented model adapters are:

- `rinalmo` for RiNALMo evaluation;
- `random` for lightweight pipeline smoke tests only.

The RiNALMo adapter currently supports the `micro` model size.

## Repository layout

```text
configs/                  YAML evaluation configurations
data/raw/ribosome/        input FASTA sequences and annotation CSV files
data/prepared/ribosome/   validated records.parquet and regions.parquet
external/                 local clones of RiNALMo and dependency-map
scripts/                  dataset preparation and evaluation entry points
src/glmfe/                datasets, model adapters, and evaluation tasks
outputs/runs/              evaluation outputs grouped by run_id
```

## Environment setup

The current environment setup script is:

```text
setup/setup_rinalmo_env.sh
```

Before running it, open the script and update `ENV_PREFIX` and `PROJECT_DIR` if
you are not using the cluster paths already written there. The remaining paths
in the script are derived from `PROJECT_DIR`.

The script:

- creates a Conda environment;
- clones or reuses the external RiNALMo repository;
- clones or reuses the external dependency-map repository;
- installs the required CUDA and FlashAttention dependencies;
- installs the local RiNALMo and dependency-map packages;
- installs Kaleido and Chrome support for dependency-map PDF export.

The RiNALMo clone command currently uses a GitHub SSH URL. If GitHub SSH access
is not configured on your machine, change that URL in the script to:

```text
https://github.com/lbcb-sci/RiNALMo.git
```

Run the setup from the repository root:

```bash
bash setup/setup_rinalmo_env.sh
```

Activate the environment using the prefix configured in the script:

```bash
conda activate /path/configured/as/ENV_PREFIX
```

The setup script installs code and runtime dependencies. It does not place a
RiNALMo checkpoint at the path used by an evaluation config. The configured
`model.rinalmo.weights` file must exist before evaluation. The current wrapper
expects that file to contain a direct PyTorch state dictionary.

## Raw dataset and preprocessing

Place ribosome input files in matching FASTA/CSV pairs:

```text
data/raw/ribosome/
  <record_id>.fasta
  <record_id>_cleaned_matches.csv
```

Each FASTA file must contain one sequence. Each annotation CSV must contain
these columns:

- `start`;
- `end`;
- `label`;
- `type`.

The preparation script currently reads `data/raw/ribosome/` and writes
`data/prepared/ribosome/`. Run it from the repository root:

```bash
python scripts/prepare_ribosome_dataset.py
```

It creates:

```text
data/prepared/ribosome/
  records.parquet
  regions.parquet
```

- `records.parquet` contains one row per sequence record, including the full
  normalized DNA sequence and basic sequence statistics.
- `regions.parquet` contains one row per annotation interval, including its
  coordinates, label, and feature type.

Prepared coordinates are 0-based, end-exclusive intervals written as
`[start, end)`.

## Running an evaluation

Copy the example before changing run-specific values:

```bash
cp configs/example.yaml configs/my_run.yaml
python scripts/run_evaluation.py configs/my_run.yaml
```

Relative paths in the configuration are resolved from the repository root.
Each run writes to:

```text
outputs/runs/<run_id>/
```

Depending on the selected tasks, the output can contain:

```text
outputs/runs/<run_id>/
  manifest.json
  config.resolved.yaml
  reconstruction/
    per_base.csv
    per_region.csv
    plots/
  dependency_maps/
    map_index.parquet
    maps/
      <record_id>/
        <map_id>.npz
        <map_id>.html
        <map_id>.pdf
```

- `manifest.json` records the run identity, model configuration, selected
  tasks, and dataset counts.
- `config.resolved.yaml` stores the configuration used for the run.
- reconstruction writes per-base and per-region result tables, with optional
  summary plots.
- dependency maps are grouped by source record and include compressed `.npz`
  arrays, standalone interactive HTML plots, and PDF plots.

## Configuration guide

### `run_id` and `overwrite`

```yaml
run_id: my-run
overwrite: false
```

`run_id` becomes the output directory name under `outputs_root`.
`overwrite: false` prevents reuse of an existing run directory.
`overwrite: true` permits writing into an existing run directory and can
replace files at the same output paths.

### `run_tasks`

```yaml
run_tasks:
  - reconstruction
  - dependency_maps
```

The supported tasks are `reconstruction` and `dependency_maps`. A run may
include either task or both.

### `dataset`

```yaml
dataset:
  dataset_id: ribosome
  prepared_dir: data/prepared/ribosome
```

- `dataset_id` is stored in output metadata.
- `prepared_dir` must contain both `records.parquet` and `regions.parquet`.

### `model`

The `adapter` field selects the model. Only the nested block with the matching
name is used.

#### Random adapter

```yaml
model:
  adapter: random

  random:
    seed: 42
    max_context_length: 1022
```

The random adapter produces normalized random outputs. It is useful for checking
the pipeline and output layout, but its scores are not scientifically
meaningful.

#### RiNALMo adapter

```yaml
model:
  adapter: rinalmo

  rinalmo:
    size: micro
    weights: models/rinalmo/rinalmo_micro_pretrained.pt
    device: cuda:0
```

- `size` currently supports only `micro`.
- `weights` is the checkpoint path relative to the repository root.
- `device` is the PyTorch device used for inference.

### `reconstruction`

```yaml
reconstruction:
  context_length: 512
  batch_size: 8
  plot_results: true
```

Reconstruction evaluates each unique annotated base once, then aggregates those
predictions back into every annotation region containing the base.

- `context_length` controls the sequence context around each target and must fit
  the selected model's maximum context length.
- `batch_size` controls model inference batching.
- `plot_results: true` writes summary plots from the generated per-region
  results.

The per-base metrics include:

- true-base probability;
- cross-entropy in bits;
- predicted base;
- reconstruction accuracy.

### `dependency_maps`

Shared dependency-map settings:

```yaml
dependency_maps:
  mode: region
  batch_size: 8
  dependency_by_masking: false
  with_reconstruction: true
```

- `mode` selects exactly one of `manual` or `region`.
- `batch_size` controls model-forward batching.
- `dependency_by_masking: false` uses nucleotide substitutions at query
  positions.
- `dependency_by_masking: true` uses masked query positions.
- `with_reconstruction: true` also stores reconstruction probabilities in each
  saved map result.

Only the configuration block selected by `mode` is used.

#### Manual mode

```yaml
dependency_maps:
  mode: manual

  manual:
    start: 4000
    end: 4100
    record_ids:
      - record-a
      - record-b
```

Manual mode evaluates the same half-open interval `[start, end)` in every listed
record. Each record ID must exactly match a `record_id` in
`records.parquet`.

#### Region mode

```yaml
dependency_maps:
  mode: region

  region:
    label: 18S_rRNA
    record_ids: all
    long_region_policy: tile
    tile_length: 512
    tile_stride: 384
```

- `label` is matched exactly against either the prepared region `label` or
  `feature_type`.
- `record_ids: all` uses regions from all matching records.
- A non-empty list of record IDs restricts the selected regions.
- A matching region that fits within the model context produces one map.
- `long_region_policy: error` stops when a selected region exceeds the model
  context length.
- `long_region_policy: tile` divides a long region into overlapping local map
  windows.
- `tile_length` must not exceed the selected model's context length.
- `tile_stride` controls the distance between regular tile starts and therefore
  the overlap between tiles.

Tiled maps capture local dependencies only. They cannot recover interactions
between positions that never occur together in the same tile.

### `outputs_root`

```yaml
outputs_root: outputs/runs
```

This path is resolved from the repository root. Each run is written below it in
a directory named by `run_id`.

## Output interpretation

- `reconstruction/per_base.csv` contains one evaluated prediction per unique
  annotated position in each record.
- `reconstruction/per_region.csv` summarizes those predictions for each
  annotation interval.
- `dependency_maps/map_index.parquet` is the authoritative table linking every
  map to its record, coordinates, source annotation, tile metadata, and output
  files.
- Dependency-map coordinates are always relative to the original full source
  record.
- For tiled region maps, `region_start` and `region_end` describe the complete
  annotation, while `tile_start` and `tile_end` describe the sequence used for
  the individual map.

## Common problems

### Missing RiNALMo checkpoint

Check that `model.rinalmo.weights` points to an existing file relative to the
repository root. The file must contain a direct PyTorch state dictionary.

### CUDA unavailable

If `device: cuda:0` fails, confirm that the machine has a CUDA-capable GPU, the
driver is available, and the configured Conda environment contains the CUDA
dependencies installed by the setup script.

### Region exceeds model context

With `long_region_policy: error`, evaluation stops when a selected annotation is
longer than the model context. Use `long_region_policy: tile` with a valid
`tile_length` and `tile_stride` to create local maps instead.

### Dependency-map PDF export fails

Run `setup/setup_rinalmo_env.sh` and use the environment it creates. The setup
installs Kaleido and the Chrome executable required by Plotly PDF export.

### Record ID does not match

Values under `manual.record_ids` or `region.record_ids` must exactly match
prepared `record_id` values. Inspect `data/prepared/ribosome/records.parquet` if
needed.

### Existing run directory

If a run directory already exists and `overwrite: false`, choose a new `run_id`
or set `overwrite: true` when replacing existing outputs is intentional.
