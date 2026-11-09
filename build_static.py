"""Export the explorer as a static site in docs/, for GitHub Pages.

    python prepare_bundle.py <dataset>   # once per dataset: bundle_<dataset>.pkl
    python build_static.py               # the site, from the bundles, data/ and saves/

DATASETS lists the datasets. Each one has its hdf5 logs in data/<dataset>/, its bundle
in bundle_<dataset>.pkl and its saved views in saves/<dataset>/. The page shows one
dataset at a time and has a menu to switch between them.

The page normally asks a Python server for its data. Here the server runs once per
dataset, on this machine: every saved view is loaded into it, which builds that
view's point set and layout exactly as the live explorer would. Then every answer the
page can ask for is written to a file under docs/api/<dataset>/:

    sets.json                      point sets
    sets/<s>/points.json           one point set's points and colors
    sets/<s>/layouts.json          its layouts
    sets/<s>/layouts/<l>.json      one layout's coordinates
    run/<id>.json                  one run's positions, for playback
    saves.json                     the saved views
    saves/load/<file>              what loading one saved view returns

docs/api/datasets.json lists the datasets. docs/static.js answers the page's requests
from these files. Fitting a new layout or building a new time window needs UMAP, so
the static page offers neither. Watershed clusters are computed in the browser, by
the same method as the server.

A dataset without saved views gets one made like TEMPLATE: the same metrics, attack
directions and display settings, fitted on its own points. It is written to its
saves folder, so later builds load it instead of fitting again.

To add a view to the site: open it in the notebook's explorer, save it, copy its JSON
from umap_explorer_saves/ into saves/<dataset>/, and run this script again.
"""
import json
import os
import pickle
import shutil
import sys
import time

import matplotlib
matplotlib.use('Agg')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features          # noqa: E402
import umap_explorer     # noqa: E402

DOCS = os.path.join(HERE, 'docs')
API = os.path.join(DOCS, 'api')
TITLE = 'Predator attack UMAP explorer'

# key: the folder name under data/ and saves/. label: the name in the page's menu.
# start: the saved view every visitor opens on. With None the page opens on the
# point set's first layout, which uses every metric and every point.
DATASETS = [
    {'key': 'prey1', 'label': 'optimized prey',
     'start': 'predator-behavior-window15_20261105-153045.json'},
    {'key': 'mass', 'label': 'predator with mass',
     'start': None},
]
TEMPLATE = ('prey1', 'predator-behavior-window15_20261105-153045.json')


def write(path, data):
    """data is bytes, as the server sends it, or anything JSON can hold"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not isinstance(data, bytes):
        data = json.dumps(data).encode()
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


def make_view_like_template(app):
    """-> the file of a new saved view: the template's layout, fitted on these points"""
    with open(os.path.join(HERE, 'saves', *TEMPLATE), encoding='utf-8') as f:
        doc = json.load(f)
    spec = doc['layout']
    meta = app.request_layout(spec['blocks'], spec['dirs'], set_id=0,
                              exclude=spec.get('exclude'))
    print(f"  fitting a layout like {TEMPLATE[1]} ...")
    while True:
        layout = next(L for L in app._layouts_meta(0) if L['id'] == meta['id'])
        if layout['status'] == 'done':
            break
        if layout['status'] == 'error':
            raise RuntimeError(f"the layout fit failed: {layout['error']}")
        time.sleep(1)
    # the selected run, its moment and the zoom belong to the template's points
    view = {k: v for k, v in doc['view'].items() if k not in ('sel', 'frame', 'view')}
    return app.save_view(doc['name'], 0, meta['id'], view)['file']


def export(ds, port):
    """-> the bytes written, after exporting one dataset to docs/api/<key>/"""
    key = ds['key']
    with open(os.path.join(HERE, f'bundle_{key}.pkl'), 'rb') as f:
        bundle = pickle.load(f)
    ctrl, res = bundle['ctrl'], bundle['res']
    for series in ctrl.values():
        for entry in series.values():
            entry['file'] = os.path.join(HERE, 'data', key, entry['file'])

    app = umap_explorer.UmapExplorer(
        res, ctrl, open_browser=False, port=port,
        feature_fn=features.controller_school_features,
        save_dir=os.path.join(HERE, 'saves', key), title=TITLE)
    try:
        start = ds['start']
        if not app.list_saves():
            make_view_like_template(app)
        saves = app.list_saves()
        loads = {}
        for save in saves:
            app.load_view(save['file'])                 # builds its set and layout
            loads[save['file']] = app.load_view(save['file'])   # now only finds them
            if loads[save['file']].get('warning'):
                print(f"  {save['file']}: {loads[save['file']]['warning']}")
        if start not in loads:
            start = None
        for got in loads.values():
            # a saved view opens with no run selected; the visitor picks one
            for k in ('sel', 'frame'):
                got['view'].pop(k, None)

        out = os.path.join(API, key)
        size = 0
        meta = app._sets_meta()
        meta.update(static=True, can_window=False, allow_delete=False, start_view=start)
        size += write(os.path.join(out, 'sets.json'), meta)
        for rec in meta['sets']:
            if rec['status'] != 'done':
                continue
            s = rec['id']
            size += write(os.path.join(out, 'sets', str(s), 'points.json'),
                          app._points_payload(s))
            layouts = [L for L in app._layouts_meta(s) if L['status'] == 'done']
            size += write(os.path.join(out, 'sets', str(s), 'layouts.json'), layouts)
            for L in layouts:
                size += write(os.path.join(out, 'sets', str(s), 'layouts', f"{L['id']}.json"),
                              app._layout_payload(s, L['id']))
        for run_id in sorted(app._files):
            size += write(os.path.join(out, 'run', f'{run_id}.json'), app._run_payload(run_id))
        size += write(os.path.join(out, 'saves.json'), {'dir': 'this site', 'saves': saves})
        for file, got in loads.items():
            size += write(os.path.join(out, 'saves', 'load', file), got)
        print(f"{key}: {len(meta['sets'])} point sets, {len(app._files)} runs, "
              f"{len(loads)} saved views, {size / 1e6:.0f} MB")
        return size
    finally:
        app.stop()


def main():
    if os.path.isdir(API):
        shutil.rmtree(API)
    size = sum(export(ds, 8095 + i) for i, ds in enumerate(DATASETS))
    size += write(os.path.join(API, 'datasets.json'),
                  {'datasets': [{'key': d['key'], 'label': d['label']} for d in DATASETS]})
    shutil.copy(os.path.join(HERE, 'umap_explorer.html'), os.path.join(DOCS, 'index.html'))
    shutil.copy(os.path.join(HERE, 'static.js'), os.path.join(DOCS, 'static.js'))
    open(os.path.join(DOCS, '.nojekyll'), 'w').close()   # serve the files as they are
    print(f"wrote docs/: {len(DATASETS)} datasets, {size / 1e6:.0f} MB of data")


if __name__ == '__main__':
    main()
