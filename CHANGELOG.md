# Changelog

Version numbers are composed of three components, i.e. _major_._minor_._patch_
As a rough guideline, patch releases are just for fixing bugs or adding minor details
(e.g. a new default parameter to some function), minor releases change slightly
larger parts of the framework or add significant new functionality (e.g. a new
optimization pipeline or support for an SQL feature). Major releases fundamentally
shift how the framework is used and indicate stability. Since we are not ready
for the 1.0 release yet, this does not matter right now.

The [history](HISTORY.md) contains the changelogs of older PostBOUND releases.

---

## Version 0.23.1

### 🐣 New features

- Added a new `WorkloadSpecificCardinalities` class. Together with
  `OfflineCardinalities`, this should provide basically the same functionality
  (and improve upon) the old `PreComputedCardinalities`.
- Using a `TextBookOptimizationPipeline` with only a cardinality estimator now
  effectively de-generates to a `MultiStageOptimizationPipeline` and eliminates
  the downsides of the textbook pipeline (mostly the emulated enumerator and the
  high overhead of cost modelling) in these cases.

### 💀 Breaking changes

None

### 📰 Updates

- Bumped DuckDB/`quacklab` to v2. This follows the 2.0 release of DuckDB and provides
  pre-built wheels on Windows.

### 🏥 Fixes

- Fixed the Postgres plan to query plan conversion sometimes producing NaN estimated
  cardinalities.

### ⚠️ Deprecations

None

### 🪲 Known bugs

- The automatic optimization of the Postgres server configuration as part of the
  Docker installation does not work on MacOS. Currently, this should be considered
  as wontfix.
- The SSB queries can currently not be loaded from the workloads module. The underlying
  data server crashed and we are currently exploring alternative, more reliable
  solutions.

### 🛣 Roadmap

The next months are going to be pretty exciting due to the imminent release of
PostgreSQL 19. As it looks like, this will be the first Postgres version that
provides official support for query hinting via the `pg_plan_advice` extension.
We will definitely add support for pg_plan_advice as a hinting backend, potentially
even making it the standard one and sunsetting development of the `pg_hint_plan`
backend. We will have to wait and see whether pg_hint_plan will be maintained in
the future.

Additionally, the `relalg` module is due for a major overhaul.
