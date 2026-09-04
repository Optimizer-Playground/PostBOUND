# PostBOUND support tools

This directory contains some additional supporting utilities for working with PostBOUND. These include:

- `setup-py-venv.sh`, a utility to install PostBOUND into a Python virtual environment. It wraps `uv sync` and performs a
  *user* installation. If you are working on PostBOUND rather than with it, use `uv sync` directly, see `CONTRIBUTING.md`
- `set-workload.sh`, a utility to update the currenctly active default Postgres database connection file
- `generate-workload.py`, a utility to generate a CSV file containing workload queries
- `ceb-generator.py`, a custom implementation of the Cardinality Estimation Benchmark generator [^1] to create workloads from
  query templates. **Currently broken**: it (and `query-generator.py`) import the removed `postbound.experiments` module and
  need to be ported or deleted
- a utility to clean the OS page cache for runtime measurements in the `drop-caches` directory

All utilities should be generally run from the repositorie's root directory, e.g. as `tools/set-workload.sh --help`.
Furthermore, Python scripts have to be run as modules, e.g. `python3 -m tools.ceb-generator --help`.

---

## References

[^1]: Parimarjan Negi et al.: "Flow-Loss: Learning Cardinality Estimates That Matter" (PVLDB 2021)
