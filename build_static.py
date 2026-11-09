"""Export the explorer as a static site in docs/, for GitHub Pages.

    python prepare_bundle.py   # once: features.py and bundle.pkl from the notebook
    python build_static.py     # the site, from bundle.pkl, data/ and saves/

The page normally asks a Python server for its data. Here the server runs once, on
this machine: every saved view in saves/ is loaded into it, which builds that view's
point set and layout exactly as the live explorer would. Then every answer the page
can ask for is written to a file under docs/:

    docs/api/sets.json                      point sets
    docs/api/sets/<s>/points.json           one point set's points and colors
    docs/api/sets/<s>/layouts.json          its layouts
    docs/api/sets/<s>/layouts/<l>.json      one layout's coordinates
    docs/api/run/<id>.json                  one run's positions, for playback
    docs/api/saves.json                     the saved views
    docs/api/saves/load/<file>              what loading one saved view returns

docs/static.js answers the page's requests from these files. Fitting a new layout or
building a new time window needs UMAP, so the static page offers neither. Watershed
clusters are computed in the browser, by the same method as the server.

To add a view to the site: open it in the notebook's explorer, save it, copy its JSON
from umap_explorer_saves/ into saves/, and run this script again.
"""
import json
import os
import pickle
import shutil
import sys

import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features          # noqa: E402
import umap_explorer     # noqa: E402

DOCS = os.path.join(HERE, 'docs')
API = os.path.join(DOCS, 'api')
START_VIEW = 'predator-behavior-window15_20261105-153045.json'
TITLE = 'Predator attack UMAP explorer'


def write(path, data):
    """data is bytes, as the server sends it, or anything JSON can hold"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not isinstance(data, bytes):
        data = json.dumps(data).encode()
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


def main():
    with open(os.path.join(HERE, 'bundle.pkl'), 'rb') as f:
        bundle = pickle.load(f)
    ctrl, res = bundle['ctrl'], bundle['res']
    for series in ctrl.values():
        for entry in series.values():
            entry['file'] = os.path.join(HERE, 'data', entry['file'])

    app = umap_explorer.UmapExplorer(
        res, ctrl, open_browser=False, port=8095,
        feature_fn=features.controller_school_features,
        save_dir=os.path.join(HERE, 'saves'), title=TITLE)
    try:
        saves = app.list_saves()
        loads = {}
        for save in saves:
            app.load_view(save['file'])                 # builds its set and layout
            loads[save['file']] = app.load_view(save['file'])   # now only finds them
            if loads[save['file']].get('warning'):
                print(f"  {save['file']}: {loads[save['file']]['warning']}")

        if os.path.isdir(API):
            shutil.rmtree(API)
        size = 0
        meta = app._sets_meta()
        meta.update(static=True, can_window=False, allow_delete=False,
                    start_view=START_VIEW if START_VIEW in loads else None)
        size += write(os.path.join(API, 'sets.json'), meta)
        for rec in meta['sets']:
            if rec['status'] != 'done':
                continue
            s = rec['id']
            size += write(os.path.join(API, 'sets', str(s), 'points.json'),
                          app._points_payload(s))
            layouts = [L for L in app._layouts_meta(s) if L['status'] == 'done']
            size += write(os.path.join(API, 'sets', str(s), 'layouts.json'), layouts)
            for L in layouts:
                size += write(os.path.join(API, 'sets', str(s), 'layouts', f"{L['id']}.json"),
                              app._layout_payload(s, L['id']))
        for run_id in sorted(app._files):
            size += write(os.path.join(API, 'run', f'{run_id}.json'), app._run_payload(run_id))
        size += write(os.path.join(API, 'saves.json'), {'dir': 'this site', 'saves': saves})
        for file, got in loads.items():
            size += write(os.path.join(API, 'saves', 'load', file), got)

        shutil.copy(os.path.join(HERE, 'umap_explorer.html'), os.path.join(DOCS, 'index.html'))
        shutil.copy(os.path.join(HERE, 'static.js'), os.path.join(DOCS, 'static.js'))
        open(os.path.join(DOCS, '.nojekyll'), 'w').close()   # serve the files as they are
        print(f"wrote docs/: {len(meta['sets'])} point sets, {len(app._files)} runs, "
              f"{len(loads)} saved views, {size / 1e6:.0f} MB of data")
    finally:
        app.stop()


if __name__ == '__main__':
    main()
