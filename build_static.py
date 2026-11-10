"""Export the explorer as a static site in docs/, for GitHub Pages.

    python prepare_bundle.py <dataset>   # once per dataset: bundle_<dataset>.pkl
    python build_static.py               # the site, from the bundles, data/ and saves/

DATASETS lists the datasets. Each one has its hdf5 logs in data/<dataset>/, its bundle
in bundle_<dataset>.pkl and its saved views in saves/<dataset>/. The page shows one
dataset at a time and has a menu to switch between them.

The page normally asks the Python explorer's web server for its data. Here the
explorer is created once per dataset, with serve=False, so no server starts. Every
saved view is loaded into it, which builds that view's point set and layout exactly
as the live explorer would. Then every answer the page can ask for is written to a
file under docs/api/<dataset>/:

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
import base64
import json
import os
import pickle
import shutil
import sys
import time

import matplotlib
import numpy as np
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
# The page opens on the first dataset in this list.
DATASETS = [
    {'key': 'mass', 'label': 'predator with mass',
     'start': None},
    {'key': 'prey1', 'label': 'optimized prey',
     'start': 'predator-behavior-window15_20261105-153045.json'},
]
TEMPLATE = ('prey1', 'predator-behavior-window15_20261105-153045.json')


def shrink_run(payload):
    """-> a run's payload in about a third of the bytes, for a faster Pages deploy

    The server sends each body's position and velocity as 32-bit floats. Here each
    position axis becomes a 16-bit integer between that axis's lowest and highest
    value in the run. The step is far below a pixel; for a run 40 length units wide
    it is 0.0006 length units. The page only draws the velocity's direction, so it
    becomes one byte per body: the angle of (first axis, last axis), in 256 steps
    of 1.4 degrees. index.html turns both back into floats.
    """
    run = json.loads(payload)
    dims = len(run['axes'])
    shape = (run['frames'], run['bodies'], dims)
    pos = np.frombuffer(base64.b64decode(run.pop('pos')), '<f4').reshape(shape).astype(float)
    vel = np.frombuffer(base64.b64decode(run.pop('vel')), '<f4').reshape(shape).astype(float)
    lo = pos.min(axis=(0, 1))
    step = np.maximum(pos.max(axis=(0, 1)) - lo, 1e-6) / 65535
    q = np.round((pos - lo) / step).astype('<u2')
    angle = np.arctan2(vel[..., -1], vel[..., 0])
    heading = np.round(angle / (2 * np.pi) * 256).astype(int) % 256
    run.update(pos_u16=base64.b64encode(q.tobytes()).decode(),
               pos_lo=lo.tolist(), pos_step=step.tolist(),
               heading_u8=base64.b64encode(heading.astype(np.uint8).tobytes()).decode())
    return json.dumps(run).encode()


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


def export(ds):
    """-> the bytes written, after exporting one dataset to docs/api/<key>/"""
    key = ds['key']
    with open(os.path.join(HERE, f'bundle_{key}.pkl'), 'rb') as f:
        bundle = pickle.load(f)
    ctrl, res = bundle['ctrl'], bundle['res']
    for series in ctrl.values():
        for entry in series.values():
            entry['file'] = os.path.join(HERE, 'data', key, entry['file'])

    app = umap_explorer.UmapExplorer(
        res, ctrl, serve=False, open_browser=False,
        feature_fn=features.controller_school_features,
        save_dir=os.path.join(HERE, 'saves', key), title=TITLE)
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
        size += write(os.path.join(out, 'run', f'{run_id}.json'),
                      shrink_run(app._run_payload(run_id)))
    size += write(os.path.join(out, 'saves.json'), {'dir': 'this site', 'saves': saves})
    for file, got in loads.items():
        size += write(os.path.join(out, 'saves', 'load', file), got)
    print(f"{key}: {len(meta['sets'])} point sets, {len(app._files)} runs, "
          f"{len(loads)} saved views, {size / 1e6:.0f} MB")
    return size


def main():
    if os.path.isdir(API):
        shutil.rmtree(API)
    size = sum(export(ds) for ds in DATASETS)
    size += write(os.path.join(API, 'datasets.json'),
                  {'datasets': [{'key': d['key'], 'label': d['label']} for d in DATASETS]})
    shutil.copy(os.path.join(HERE, 'umap_explorer.html'), os.path.join(DOCS, 'index.html'))
    shutil.copy(os.path.join(HERE, 'static.js'), os.path.join(DOCS, 'static.js'))
    open(os.path.join(DOCS, '.nojekyll'), 'w').close()   # serve the files as they are
    print(f"wrote docs/: {len(DATASETS)} datasets, {size / 1e6:.0f} MB of data")


if __name__ == '__main__':
    main()
