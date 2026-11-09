## Rebuilding the site

The site holds several datasets. `DATASETS` in `build_static.py` lists them. Each
dataset has a short key, such as `prey1`. Its hdf5 logs are in `data/<key>/`, and its
saved views are in `saves/<key>/`.

1. `python prepare_bundle.py <key>` reads `data/<key>/` and the notebook, and writes
   `features.py` and `bundle_<key>.pkl`. It takes about 10 to 20 minutes per dataset.
   Run it again only when that dataset or the notebook's feature code changes.
2. `python build_static.py` writes `docs/` for all datasets.

To add a view, save it in the notebook's explorer, copy its JSON file from
`evo_analysis/umap_explorer_saves/` into `saves/<key>/`, and run step 2 again.

To add a dataset, copy its hdf5 files into `data/<key>/`, add an entry to
`DATASETS`, and run both steps. A dataset with no saved views gets one made like
`TEMPLATE`, with the same metrics and settings.

The page opens on the first dataset. A menu in the header switches to another one.
A link to one dataset looks like `index.html?data=mass`.

To try the site locally, run `python -m http.server 8000` inside `docs/` and open
http://localhost:8000.

`data/` and the `bundle_*.pkl` files are build inputs and are not committed.
