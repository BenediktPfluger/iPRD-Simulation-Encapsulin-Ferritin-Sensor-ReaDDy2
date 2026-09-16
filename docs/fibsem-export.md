# FIB-SEM comparison export

Exporting a finished run's final frame into the FIB-SEM segmentation schema, so a simulation
drops straight into the experimental analysis pipeline. Driven by
`Export_for_FIB-SEM_Comparison.ipynb` and implemented in `qtft/fibsem_export.py`.

---

To compare a simulated end state against **FIB-SEM** segmentation data, `qtft.fibsem_export`
writes the **final frame** in the same schema the segmentation pipeline produces, so a
simulation drops straight into the existing experimental analysis. It is **read-only**: it
consumes a finished trajectory and never builds or runs a simulation.

Driven from `Export_for_FIB-SEM_Comparison.ipynb` (a settings cell, a run cell, and a
cluster-coloured scatter as a visual unwrap check), or directly:

```python
from qtft.config import SimulationConfig
from qtft import fibsem_export

traj, cfg_path = fibsem_export.find_run_files("Simulation_Files_Single_Runs/<run_dir>")
config = SimulationConfig.load_json(cfg_path)
df, info = fibsem_export.export(traj, config, voxel_nm=4.0)
```

`find_run_files` resolves the run layouts used here: a plain run (`trajectory.h5`), a phased
agglomeration↔deagglomeration run (`trajectory_combined.h5`, else the last
`phase_NNN/trajectory.h5`), and the matching config (`<param_string>_config.json`, or
`ensemble_config.json` for an ensemble replica).

**Where the files go.** By default the export writes into
`<run_dir>/FIBSEM_Comparison_Export/` — beside the run's own `.h5`, `.xyz` and
`_config.json`, mirroring the `Plots/` convention of the plotting outputs — so exports from different runs
never overwrite one another. `fibsem_export.default_out_dir(traj)` returns that path (it
steps up out of a `phase_NNN/` folder, so a phased run's export lands at run level, next to
`Plots/`). Pass `out_dir` explicitly to `export` to write somewhere else instead.

**What it extracts.**

- **Encapsulins only.** Qt (free) and QtC (bound) are exported; ferritin is sub-resolution in
  FIB-SEM and is dropped — but it still takes part in the periodic unwrap, since it defines
  the Qt–Ft–Qt connectivity of a cluster.
- **Clusters are ground truth**, taken from ReaDDy's topology graph — one topology = one
  cluster, so no DBSCAN is needed on the simulation side. Because every particle is placed as
  its own single-particle topology (`engine.place_particles`), boundness is read off the
  graph: a topology with more than one particle means its encapsulin is bonded (a Qt–Ft dimer
  counts), while a size-1 topology is an unbound encapsulin, i.e. a FIB-SEM singleton.
- **Positions are PBC-unwrapped per cluster**, then shifted so the minimum corner sits at the
  origin (all coordinates ≥ 0).
- **Volumes are analytical** (4/3·π·r³), and `radius_nm` is stored per row so a later notebook
  can compute an exact mass integral ⟨M(R)⟩ from ball–ball intersections, with no voxelisation.

**Outputs** (into `<run_dir>/FIBSEM_Comparison_Export/` unless `out_dir` overrides it,
suffixed with `file_tag`, default `_simulation`):

| File | Format | Contents |
|------|--------|----------|
| `encapsulin_centroids_simulation.csv` | CSV | one row per encapsulin: `label, z/y/x_nm, z/y/x_vox, radius_nm, volume_nm3, cluster, is_clustered` |
| `structural_information_and_metadata_simulation.json` | JSON | source trajectory, final step / µs, encapsulin and cluster counts, cluster-size histogram, box, applied coordinate offset, and the full flattened config |

> **Caveat — periodicity is not preserved.** Each cluster is unwrapped independently and then
> everything is shifted by one global offset. Within a cluster the geometry is exact, but
> between clusters periodicity is gone: a cluster unwrapped past the boundary can end up
> spatially overlapping another, and the bounding volume exceeds the true box. That is
> harmless for per-cluster shape/size statistics, but it biases any metric that samples
> neighbourhoods *across* clusters (⟨M(R)⟩, RDF). Decide deliberately whether such a metric
> should be computed under PBC before the unwrap.
