## Rebuilding the site

1. `python prepare_bundle.py` reads `data/` and the notebook, and writes
   `features.py` and `bundle.pkl`. It takes about 10 minutes. Run it again only
   when the data or the notebook's feature code changes.
2. `python build_static.py` writes `docs/` from the saved views in `saves/`.

To add a view, save it in the notebook's explorer, copy its JSON file from
`evo_analysis/umap_explorer_saves/` into `saves/`, and run step 2 again.

To try the site locally, run `python -m http.server 8000` inside `docs/` and open
http://localhost:8000.

`data/` and `bundle.pkl` are build inputs and are not committed.
