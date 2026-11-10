"""Interactive explorer for a controller UMAP: click a point, watch its run.

    import umap_explorer
    app = umap_explorer.UmapExplorer(res, ctrl, folder_path)   # opens a browser tab
    app.stop()

`res` is the dict returned by controller_umap or controller_school_umap, and `ctrl`
the data dict it was computed from. The embedding is sent to the page once. A run's
raw positions are read from its hdf5 log only when one of its points is clicked, so
the page stays light however many runs the embedding holds.

The page can also lay the same points out again from a subset of the metrics. Every
block of res['X'] was standardised and balanced on its own, so a layout over some
blocks is UMAP on just their columns, with the original UMAP settings. Each new
layout is rotated, mirrored and scaled onto the original one, so the two can be
compared point by point. The fits run one at a time in a background thread.

A layout can also be restricted to some attack directions. Then every column is
standardised again over those points alone, which gives exactly the features that
controller_umap or controller_school_umap builds from data holding only those
directions; see _restandardise.

Pass the feature function as well, and the page can build whole new point sets from
short stretches of time:

    app = umap_explorer.UmapExplorer(res, ctrl, folder_path,
                                     feature_fn=controller_school_features)

A point set of window w stacks w consecutive iterations into every point and takes
points one window apart, as feature_fn(data, window=w, window_stat='stack',
stride='window') does. Each point set has its own metric and direction layouts.

A layout can also keep each point's neighbours away from its own run. UMAP builds
its picture from every point's nearest neighbours, and consecutive moments of one
run are each other's nearest neighbours, so the runs come out as paths. Built from
a neighbour list without the point's own run, or without the moments of its run
closer than a gap, similar moments of different runs pull together instead; see
_excluded_knn.

A view can be saved to disk and loaded again in a later session: the point set, the
layout with its coordinates and settings, and what the page showed, such as the
colors, the selected run and moment and the zoom. Saves are JSON files in save_dir,
by default umap_explorer_saves next to the notebook. Loading checks that the saved
points are the same runs and iterations as the explorer's own, and needs no UMAP fit.

The server binds to 127.0.0.1 only and runs in a daemon thread, so the notebook stays
usable while it serves, and it goes away with the kernel.

This copy is the one the Hugging Face Space runs, through app.py. There it binds to
host='0.0.0.0', opens every visitor on start_view, and allow_delete=False keeps the
visitors from deleting saved views.
"""
import base64
import datetime
import glob
import hashlib
import json
import os
import queue
import re
import threading
import time
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import h5py
import numpy as np

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'umap_explorer.html')

# port -> explorer, so re-running a notebook cell replaces the old one. importlib.reload
# re-runs this file in the same namespace, so the servers started before a reload are
# kept here and still get stopped, instead of holding their port with stale code.
_RUNNING = globals().get('_RUNNING', {})

# the page checks this, and says so when it is served by an older version of this file
API_VERSION = 7

# the window sizes the page offers, in iterations
WINDOW_RANGE = (2, 60)


def _f32(array):
    """-> base64 of a float32 array; NaN survives, unlike in JSON"""
    return base64.b64encode(np.ascontiguousarray(array, dtype='<f4').tobytes()).decode()


def _i32(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype='<i4').tobytes()).decode()


def _u8(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).decode()


def _attr(value):
    """-> an hdf5 attribute as a plain python value"""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _procrustes(Y_fit, R_fit):
    """-> (rotation, scale, mu_y, mu_r) that best maps the rows of Y_fit onto R_fit"""
    Y_fit, R_fit = np.asarray(Y_fit, dtype=float), np.asarray(R_fit, dtype=float)
    mu_y, mu_r = Y_fit.mean(axis=0), R_fit.mean(axis=0)
    Yc, Rc = Y_fit - mu_y, R_fit - mu_r
    U, s, Vt = np.linalg.svd(Yc.T @ Rc)
    return U @ Vt, s.sum() / max(np.sum(Yc ** 2), 1e-12), mu_y, mu_r


def _align(Y, reference):
    """-> Y rotated, possibly mirrored, scaled and shifted onto reference

    UMAP's axes carry no meaning, so two layouts of the same points can differ by a
    rotation or a mirror image alone. Orthogonal Procrustes removes exactly that and
    nothing else, so whatever difference is left is a real change in the layout.
    """
    Q, scale, mu_y, mu_r = _procrustes(Y, reference)
    return (np.asarray(Y, dtype=float) - mu_y) @ Q * scale + mu_r


def _restandardise(X, scale):
    """-> X with every column standardised over these rows, then rescaled by scale

    Each column of res['X'] is (raw - mean) / std, divided by sqrt(block width) when
    block_balance was on, with mean and std pooled over every point. Standardising
    gives the same result for raw and for any shift and positive rescaling of it, so
    standardising the stored column over a subset of rows gives exactly what the raw
    column standardised over that subset would. Multiplying by the column's std over
    all rows, which is 1 / sqrt(width) or 1, puts the block balance back without
    needing to know whether it was on. A column that is constant within the subset is
    zeroed, as the feature builders do.
    """
    std = X.std(axis=0)
    out = (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0) * scale
    out[:, std == 0] = 0.0
    return out


def _neighbours_kept(Y, reference, k=15):
    """-> the mean share of each point's k nearest neighbours in reference that are
    also among its k nearest neighbours in Y. Both hold the same points, in the same
    order, so for a layout of one attack direction the original layout is compared
    over that direction's points only."""
    from sklearn.neighbors import NearestNeighbors
    # kneighbors() without a query leaves each point itself out of its own list
    k = min(k, len(Y) - 1)
    ref = NearestNeighbors(n_neighbors=k).fit(reference).kneighbors(return_distance=False)
    new = NearestNeighbors(n_neighbors=k).fit(Y).kneighbors(return_distance=False)
    shared = sum(len(np.intersect1d(a, b, assume_unique=True)) for a, b in zip(ref, new))
    return float(shared / (k * len(Y)))


def _excluded_knn(X, ids, iterations, k, gap=None, metric='euclidean'):
    """-> (indices, distances), each (n, k), for UMAP's precomputed_knn

    Row i holds point i itself first, as UMAP's own neighbour search does, then its
    k - 1 nearest points that are not excluded. gap=None excludes every point of the
    same run; a gap in iterations excludes only the moments of the same run fewer than
    gap iterations away. The search widens until every row has enough neighbours.
    """
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, dtype=float)
    n, need = len(X), k - 1
    _, run = np.unique(np.asarray(ids), return_inverse=True)
    iterations = np.asarray(iterations)
    tree = NearestNeighbors(metric=metric).fit(X)
    out_i, out_d = np.empty((n, k), dtype=np.int64), np.empty((n, k), dtype=float)
    out_i[:, 0], out_d[:, 0] = np.arange(n), 0.0
    todo, q = np.arange(n), min(n, 2 * k + 10)
    while len(todo):
        dist, idx = tree.kneighbors(X[todo], n_neighbors=q)
        excluded = (idx == todo[:, None]) | (run[idx] == run[todo][:, None])
        if gap is not None:
            excluded &= ((idx == todo[:, None])
                         | (np.abs(iterations[idx] - iterations[todo][:, None]) < gap))
        done = (~excluded).sum(axis=1) >= need
        # a stable sort puts the allowed neighbours first, still nearest first
        pick = np.argsort(excluded, axis=1, kind='stable')[:, :need]
        rows = todo[done]
        out_i[rows, 1:] = np.take_along_axis(idx, pick, axis=1)[done]
        out_d[rows, 1:] = np.take_along_axis(dist, pick, axis=1)[done]
        todo = todo[~done]
        if len(todo) and q >= n:
            raise ValueError(f"{len(todo)} points have fewer than {need} neighbours "
                             "outside the excluded ones")
        q = min(n, 2 * q)
    return out_i, out_d


def _same_run_share(X, ids, k=15):
    """-> the share of each point's k nearest neighbours in X that come from its own
    run; the same measure as same_run_share in controller_umap.ipynb"""
    from sklearn.neighbors import NearestNeighbors
    _, run = np.unique(np.asarray(ids), return_inverse=True)
    k = min(k, len(run) - 1)
    nn = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(return_distance=False)
    return float(np.mean(run[nn] == run[:, None]))


WSHED_GRID = 400          # density map pixels per side
WSHED_SIGMAS = np.geomspace(0.003, 0.15, 45)   # smoothing widths, as fractions of the map width
WSHED_FLOOR = 1e-5        # below this density a pixel belongs to no region, as in the paper
WSHED_MAX = 60            # the most regions the page asks for


class _DensityMap:
    """the smoothed point density of one 2D layout, on a square grid

    As in MotionMapper's findPointDensity: a 2D histogram of the points, summing to 1,
    convolved with a Gaussian whose peak is 1. A lone point then has density 1/N at
    its centre. The convolution runs through one padded FFT, so trying many widths
    costs one inverse FFT each.
    """

    def __init__(self, Y, grid=WSHED_GRID):
        lo, hi = Y.min(axis=0), Y.max(axis=0)
        self.span = float(max(hi - lo)) or 1.0
        centre = (lo + hi) / 2
        half = 0.55 * self.span                    # a 10% margin on every side
        self.edges = [np.linspace(c - half, c + half, grid + 1) for c in centre]
        self.pixel = 2 * half / grid
        H = np.histogram2d(Y[:, 0], Y[:, 1], bins=self.edges)[0]
        H /= H.sum()
        self.grid, self.size = grid, 2 * grid      # padded, so the blur never wraps
        self._fH = np.fft.rfft2(H, s=(self.size, self.size))
        f0 = np.fft.fftfreq(self.size)[:, None]
        f1 = np.fft.rfftfreq(self.size)[None, :]
        self._f2 = f0 ** 2 + f1 ** 2

    def density(self, sigma):
        """-> (grid, grid) density for a Gaussian of std sigma, in layout units"""
        s = sigma / self.pixel
        kernel = np.exp(-2 * np.pi ** 2 * s ** 2 * self._f2) * (2 * np.pi * s ** 2)
        D = np.fft.irfft2(self._fH * kernel, s=(self.size, self.size))[:self.grid, :self.grid]
        return np.maximum(D, 0)

    def regions(self, sigma):
        """-> (labels, density): the watershed of the inverse density"""
        from skimage.segmentation import watershed
        D = self.density(sigma)
        labels = watershed(-D, connectivity=2)
        labels[D < WSHED_FLOOR] = 0
        return labels, D


def _plateaus(fractions, counts, top=4):
    """-> the region counts that hold over the widest range of smoothing, widest first

    Each entry is (count, smallest fraction, largest fraction, width in log units).
    One region is left out, since it says nothing.
    """
    runs, start = [], 0
    for i in range(1, len(counts) + 1):
        if i == len(counts) or counts[i] != counts[start]:
            if counts[start] >= 2:
                lo, hi = fractions[start], fractions[i - 1]
                # a run's width spans half a step past each end, so a lone step counts
                step = np.log(fractions[1] / fractions[0])
                runs.append((int(counts[start]), float(lo), float(hi),
                             float(np.log(hi / lo) + step)))
            start = i
    best = {}
    for run in runs:                 # a count may come back; keep its widest run
        if run[0] not in best or run[3] > best[run[0]][3]:
            best[run[0]] = run
    return sorted(best.values(), key=lambda r: -r[3])[:top]


def _colour_regions(labels, n, colours=8):
    """-> a colour index per region 1..n, so regions that touch differ where possible

    Greedy colouring of the adjacency graph, largest-degree first. Colours only keep
    neighbours apart; the region numbers on the map name the regions.
    """
    pairs = set()
    for a, b in ((labels[1:, :], labels[:-1, :]), (labels[:, 1:], labels[:, :-1])):
        m = (a != b) & (a > 0) & (b > 0)
        for p, q in zip(a[m].tolist(), b[m].tolist()):
            pairs.add((min(p, q), max(p, q)))
    adj = {r: set() for r in range(1, n + 1)}
    for p, q in pairs:
        adj[p].add(q); adj[q].add(p)
    colour = {}
    for r in sorted(adj, key=lambda r: -len(adj[r])):
        used = {colour[q] for q in adj[r] if q in colour}
        free = [c for c in range(colours) if c not in used]
        colour[r] = free[0] if free else min(range(colours),
                                             key=lambda c: sum(colour.get(q) == c for q in adj[r]))
    return [colour[r] for r in range(1, n + 1)]


def _boundaries(labels, edges):
    """-> flat [x0, y0, x1, y1, ...] of the pixel edges between different labels

    Runs of equal edges along a row or column are merged into one segment.
    """
    ex, ey = edges
    out = []
    # edges between columns i and i+1 (labels[i, j] is x bin i, y bin j): vertical lines
    diff = labels[1:, :] != labels[:-1, :]
    for i in np.flatnonzero(diff.any(axis=1)):
        j = np.flatnonzero(diff[i])
        breaks = np.flatnonzero(np.diff(j) != 1)
        for a, b in zip(np.r_[0, breaks + 1], np.r_[breaks, len(j) - 1]):
            out += [ex[i + 1], ey[j[a]], ex[i + 1], ey[j[b] + 1]]
    diff = labels[:, 1:] != labels[:, :-1]
    for j in np.flatnonzero(diff.any(axis=0)):
        i = np.flatnonzero(diff[:, j])
        breaks = np.flatnonzero(np.diff(i) != 1)
        for a, b in zip(np.r_[0, breaks + 1], np.r_[breaks, len(i) - 1]):
            out += [ex[i[a]], ey[j + 1], ex[i[b] + 1], ey[j + 1]]
    return np.asarray(out, dtype=np.float32)


def _decode(text, dtype):
    """-> the array that _f32 or _i32 encoded"""
    return np.frombuffer(base64.b64decode(text), dtype=dtype).copy()


def _fingerprint(X):
    """-> a short hash of a feature matrix, to notice when a save was made from other
    features than the ones loaded now"""
    if X is None:
        return None
    return hashlib.sha1(np.ascontiguousarray(np.round(X, 6), dtype='<f8').tobytes()).hexdigest()[:16]


def _jsonable(value):
    """-> value with numpy scalars and other odd types made JSON-safe"""
    return json.loads(json.dumps(value, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))


def _strip(record):
    """-> a layout's or point set's metadata, without its arrays"""
    return {k: v for k, v in record.items() if not k.startswith('_') and k not in ('x', 'y')}


class _Server(ThreadingHTTPServer):
    """ThreadingHTTPServer that never shares its port.

    The stock server sets SO_REUSEADDR. On Windows that lets a second server bind a
    port that is already serving, and the old server keeps answering, so a re-run
    cell would silently show stale code. Without it, a busy port fails to bind and
    the next one is tried.
    """
    allow_reuse_address = os.name != 'nt'
    daemon_threads = True


def _stop_stale_servers(port):
    """stop every explorer server in this process that still listens on port

    Earlier versions of this file lost track of their servers on importlib.reload,
    so such servers can only be found by looking through the live objects.
    """
    import gc
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, ThreadingHTTPServer):
                continue
            handler = getattr(obj, 'RequestHandlerClass', None)
            if 'UmapExplorer' not in getattr(handler, '__qualname__', ''):
                continue
            if obj.server_address[1] != port or obj.socket.fileno() == -1:
                continue
        except Exception:
            continue
        # shutdown() waits for serve_forever to notice, so never let it block for long
        stopper = threading.Thread(target=obj.shutdown, daemon=True)
        stopper.start()
        stopper.join(timeout=2)
        obj.server_close()
        print(f"stopped an older UMAP explorer server on port {port}")


# blocks the page offers one component at a time, and the names of their components
SPLIT_BLOCKS = {'prey_dir': ('forward', 'right', 'up'),
                'com_dir': ('forward', 'right', 'up')}


def _metric_choices(blocks, live):
    """-> [(name, columns, live columns)], the metrics the page can tick on and off

    Most blocks are one choice. The blocks in SPLIT_BLOCKS become one choice per
    body-frame component, named like 'prey_dir[right]', so a layout can leave out one
    component and keep the other. A component's columns keep the weight its block
    gave them, so ticking every component gives back the block exactly. Within a
    block the columns run component by component and, in a stacked window, frame by
    frame, so component j is every d-th column starting at j. A component that never
    varies, such as 'up' in a 2d run, is not offered, since it carries nothing.
    """
    out = []
    for name, where in sorted(blocks.items(), key=lambda kv: kv[1].start):
        cols = np.arange(where.start, where.stop)
        axes = SPLIT_BLOCKS.get(name)
        if axes and len(cols) % len(axes) == 0:
            d = len(axes)
            for j, axis in enumerate(axes):
                part = cols[j::d]
                if live[part].any():
                    out.append((f"{name}[{axis}]", part, int(live[part].sum())))
        else:
            out.append((name, cols, int(live[cols].sum())))
    return out


class _PointSet:
    """One embedded set of points, its feature matrix and the layouts fitted on it"""

    def __init__(self, set_id, res, window=None, seconds=None):
        self.id, self.window = set_id, window
        ids = np.asarray(res['ids'])
        embedding = np.asarray(res['embedding'], dtype=float)
        self.res, self.ids, self.iterations = res, ids, np.asarray(res['iterations'])
        self.reference = embedding
        self.dirs = np.asarray(res['attack_dirs'])
        self.dir_names = list(dict.fromkeys(res['attack_dirs']))
        blocks, X = res.get('blocks') or {}, res.get('X')
        if X is None or not blocks:   # an older result: the page hides the panel
            self.X, self.choices, self.block_info = None, [], []
        else:
            # float64, so restandardising a subset matches a fresh build to rounding
            self.X = np.asarray(X, dtype=float)
            self.scale = self.X.std(axis=0)
            self.choices = _metric_choices(blocks, self.scale > 0)
            self.block_info = [(name, len(cols), live) for name, cols, live in self.choices]
        self.layouts = [{
            'id': 0, 'blocks': [name for name, _, _ in self.choices], 'dirs': None,
            'exclude': None, 'same_run_2d': _same_run_share(embedding, ids),
            'status': 'done', 'original': True, 'points': int(len(embedding)),
            'columns': int(self.X.shape[1]) if self.X is not None else None,
            'live': int(sum(info[2] for info in self.block_info)),
            'kept': 1.0, 'seconds': seconds, 'error': None,
            'x': embedding[:, 0], 'y': embedding[:, 1],
        }]
        self._points = None

    def layout_matrix(self, rows, columns, dirs):
        """-> the feature matrix a layout is fitted on"""
        X = self.X[np.ix_(rows, columns)]
        return X if dirs is None else _restandardise(X, self.scale[columns])


class UmapExplorer:
    """A local web page over one UMAP result.

    folder_path is only needed when `data` was extracted before the run files were
    recorded in it. In that case the files are found again with the same glob that
    extract_data_from_folder uses, and every run is checked against its stored
    predator trajectory before it is shown, so a mismatched file cannot be played
    silently.

    start_idx must match the extraction: the first start_idx iterations of every
    log are skipped there, and frame 0 of the player is iteration start_idx of the
    file, so frame t and the embedding's iteration t are the same moment. Data
    extracted with the current extract_data_from_folder records it, and None takes
    it from there, falling back to that function's default of 42.

    feature_fn, when given, lets the page build point sets over windows of its own
    choosing. It is called as feature_fn(data, window=w, window_stat='stack',
    stride='window', **feature_kwargs) and must return what controller_features and
    controller_school_features return. Pass the same function, and the same
    feature_kwargs, that res was built with, so the point sets differ in the window
    alone.

    save_dir is where the page's saved views go, umap_explorer_saves in the current
    directory unless given.

    serve=False starts no web server. The explorer's methods still work, which is
    how build_static.py exports the static site.
    """

    def __init__(self, res, data, folder_path=None, port=8050, open_browser=True,
                 start_idx=None, file_prefix="simulation_positions", title=None,
                 cache_runs=24, feature_fn=None, feature_kwargs=None, save_dir=None,
                 host='127.0.0.1', start_view=None, allow_delete=True, serve=True):
        self.data = data
        self.host, self.start_view, self.allow_delete = host, start_view, allow_delete
        self.save_dir = os.path.abspath(save_dir or os.path.join(os.getcwd(),
                                                                 'umap_explorer_saves'))
        if start_idx is None:
            entry = next(iter(next(iter(data.values())).values()))
            start_idx = entry.get('start_idx', 42)
        self.start_idx = start_idx
        self.title = title or "controller UMAP explorer"
        self.feature_fn, self.feature_kwargs = feature_fn, dict(feature_kwargs or {})
        self.output_step = float(self.feature_kwargs.get('output_step', 0.1))
        self._files = self._run_files(data, folder_path, file_prefix)
        self._cache, self._cache_size = OrderedDict(), cache_runs
        self._lock = threading.Lock()       # the run cache
        self._layout_lock = threading.Lock()  # point sets and layouts
        self._wshed_lock = threading.Lock()   # the density maps, one per layout
        self._wshed = {}                      # (set id, layout id) -> sweep over smoothing
        self._jobs = None
        reducer = res.get('reducer')
        self._umap_kwargs = {
            'n_neighbors': getattr(reducer, 'n_neighbors', 50),
            'min_dist': getattr(reducer, 'min_dist', 0.0),
            'metric': getattr(reducer, 'metric', 'euclidean'),
            'random_state': getattr(reducer, 'random_state', 42),
        }
        first = _PointSet(0, res)
        self._sets = [{'id': 0, 'window': None, 'original': True, 'status': 'done',
                       'points': int(len(first.ids)),
                       'columns': first.layouts[0]['columns'],
                       'same_run': None, 'seconds': None, 'error': None,
                       '_set': first}]
        self.res = res

        # serve=False gives the explorer's data without a web server, for build_static.py
        self._server, self.port = None, None
        if not serve:
            return
        if port in _RUNNING:
            _RUNNING.pop(port).stop()
        _stop_stale_servers(port)
        self._server = None
        for candidate in range(port, port + 20):
            try:
                self._server = _Server((host, candidate), self._handler())
                break
            except OSError:
                continue
        if self._server is None:
            raise OSError(f"no free port in {port}-{port + 19}")
        self.port = self._server.server_address[1]
        _RUNNING[self.port] = self

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"UMAP explorer running at {self.url}  ({len(res['ids'])} points, "
              f"{len(set(np.asarray(res['ids']).tolist()))} runs) - app.stop() ends it"
              + ("" if feature_fn else "\n  pass feature_fn=controller_school_features "
                 "to choose the time window on the page"))
        if open_browser:
            webbrowser.open(self.url)

    @property
    def url(self):
        shown = '127.0.0.1' if self.host in ('', '0.0.0.0') else self.host
        return f"http://{shown}:{self.port}/"

    def stop(self):
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            _RUNNING.pop(self.port, None)
            self._server = None
            print(f"UMAP explorer on port {self.port} stopped")

    def __repr__(self):
        state = 'running at ' + self.url if self._server else 'stopped'
        return f"<UmapExplorer {state}>"

    def point_set(self, set_id):
        """-> the result dict of a point set, in the same form as res, for the notebook"""
        ps = self._get_set(set_id)
        return dict(ps.res, embedding=ps.reference)

    # -- data ---------------------------------------------------------------------

    @staticmethod
    def _run_files(data, folder_path, file_prefix):
        """-> {run id: hdf5 path}"""
        entries = next(iter(data.values()))
        if all('file' in entry for entry in entries.values()):
            return {i: entry['file'] for i, entry in entries.items()}
        if folder_path is None:
            raise ValueError("this data was extracted before run files were recorded "
                             "in it; pass folder_path, or re-extract")
        # the same call, in the same order, as extract_data_from_folder: a run id is
        # the position of its file in this list
        files = glob.glob(os.path.join(folder_path, f"{file_prefix}*"))
        return {i: files[i] for i in entries if i < len(files)}

    def _get_set(self, set_id):
        with self._layout_lock:
            if not 0 <= set_id < len(self._sets):
                raise KeyError(f"no point set {set_id}")
            record = self._sets[set_id]
            if record['status'] != 'done':
                raise KeyError(f"point set {set_id} is {record['status']}")
            return record['_set']

    def _sets_meta(self):
        with self._layout_lock:
            return {'api_version': API_VERSION, 'can_window': self.feature_fn is not None,
                    'window_range': list(WINDOW_RANGE), 'output_step': self.output_step,
                    'start_view': self.start_view, 'allow_delete': self.allow_delete,
                    'sets': [_strip(record) for record in self._sets]}

    def _points_payload(self, set_id):
        ps = self._get_set(set_id)
        if ps._points is None:
            ps._points = self._points_fields(ps)
        # the layouts are added fresh every time, so a layout fitted after the first
        # visit to this point set is listed too
        return json.dumps(dict(ps._points, layouts=self._layouts_meta(set_id))).encode()

    def _points_fields(self, ps):
        """-> the parts of a point set's payload that never change"""
        set_id = ps.id
        res = ps.res
        ids = ps.ids
        dir_index = {d: k for k, d in enumerate(ps.dir_names)}
        features = res.get('features', {})
        runs = {}
        for i in sorted(set(ids.tolist())):
            path = self._files.get(i)
            runs[str(i)] = {'file': os.path.basename(path) if path else None,
                            'n': int(np.sum(ids == i))}
        return {
            'api_version': API_VERSION,
            'set': set_id,
            'window': ps.window,
            'title': self.title,
            'n_features': int(ps.X.shape[1]) if ps.X is not None else None,
            'x': _f32(ps.reference[:, 0]),
            'y': _f32(ps.reference[:, 1]),
            'run': _i32(ids),
            'it': _i32(ps.iterations),
            'progress': _f32(res['progress']),
            'pass_time': _f32(res['pass_time']) if 'pass_time' in res else None,
            'dir': _i32([dir_index[d] for d in res['attack_dirs']]),
            'dirs': ps.dir_names,
            'features': {name: _f32(values) for name, values in features.items()},
            'runs': runs,
            'blocks': [{'name': name, 'width': width, 'live': live}
                       for name, width, live in ps.block_info],
        }

    def _run_payload(self, run_id):
        with self._lock:
            if run_id in self._cache:
                self._cache.move_to_end(run_id)
                return self._cache[run_id]
        path = self._files.get(run_id)
        if path is None:
            raise KeyError(f"no file is known for run {run_id}")

        with h5py.File(path, 'r') as f:
            attrs = {key: _attr(f.attrs[key]) for key in f.attrs}
            keys = sorted((k for k in f.keys() if k.startswith('iteration_')),
                          key=lambda k: int(k.split('_')[1]))[self.start_idx:]
            if not keys:
                raise ValueError(f"{os.path.basename(path)} has no iterations past "
                                 f"start_idx={self.start_idx}")
            positions = np.stack([f[k]['positions'][:] for k in keys])
            velocities = np.stack([f[k]['velocities'][:] for k in keys])
            flee = (np.stack([np.linalg.norm(f[k]['flee_force'][:], axis=1) > 0
                              for k in keys])
                    if 'flee_force' in f[keys[0]] else None)

        self._check_alignment(run_id, positions, path)

        # the prey swim in the y=0 plane in a 2d run, so only x and z are sent then
        planar = (str(attrs.get('use_2d', '')).lower() == 'true'
                  or np.ptp(positions[..., 1]) == 0)
        axes = [0, 2] if planar else [0, 1, 2]
        n_frames, n_bodies = positions.shape[:2]
        payload = json.dumps({
            'id': run_id,
            'file': os.path.basename(path),
            'attack_dir': attrs.get('predator_init_dir'),
            'frames': n_frames,
            'bodies': n_bodies,           # the last body is the predator
            'axes': ['xyz'[a] for a in axes],
            'pos': _f32(positions[..., axes]),
            'vel': _f32(velocities[..., axes]),
            'flee': _u8(flee[:, :-1]) if flee is not None else None,
            'output_step': float(attrs.get('output_step', 0.1)),
            'r_catch': float(attrs.get('predator_r_catch', 1.0)),
            'start_idx': self.start_idx,
        }).encode()

        with self._lock:
            self._cache[run_id] = payload
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return payload

    def _check_alignment(self, run_id, positions, path):
        """refuse to play a file whose predator path disagrees with the stored one"""
        if 'trajectory' not in self.data or run_id not in self.data['trajectory']:
            return
        stored = np.asarray(self.data['trajectory'][run_id]['values'], dtype=float)
        n = min(len(stored), len(positions))
        probe = np.unique(np.linspace(0, n - 1, min(n, 5)).astype(int))
        if not np.allclose(positions[probe, -1, :], stored[probe], atol=1e-3):
            raise ValueError(f"{os.path.basename(path)} does not match run {run_id}'s "
                             "stored predator trajectory; check folder_path and "
                             "start_idx, or re-extract")

    # -- background work: point sets and layouts, one at a time --------------------

    def _submit(self, job):
        if self._jobs is None:
            self._jobs = queue.Queue()
            threading.Thread(target=self._worker, daemon=True).start()
        self._jobs.put(job)

    def _worker(self):
        while True:
            job = self._jobs.get()
            if job[0] == 'set':
                self._build_set(job[1])
            else:
                self._fit_layout(*job[1:])

    def _umap(self, X, exclude=None, ids=None, iterations=None):
        """UMAP with the explorer's settings. exclude, when set, builds the neighbour
        list without each point's own run ('run') or without the moments of its run
        closer than that many seconds"""
        import umap
        kwargs = dict(self._umap_kwargs)
        kwargs['n_neighbors'] = k = min(kwargs['n_neighbors'], len(X) - 1)
        if exclude is not None:
            gap = None if exclude == 'run' else max(1, int(round(exclude / self.output_step)))
            kwargs['precomputed_knn'] = _excluded_knn(X, ids, iterations, k, gap,
                                                      kwargs['metric'])
            # below 4096 points UMAP would compute all distances itself and ignore
            # the list, unless it is told to take its usual neighbour-graph path
            kwargs['force_approximation_algorithm'] = True
        import warnings
        with warnings.catch_warnings():
            # a precomputed list has no search index, so UMAP warns that transform()
            # on new data is unavailable, which the explorer never uses
            warnings.filterwarnings('ignore', message='precomputed_knn')
            return umap.UMAP(n_components=2, **kwargs).fit_transform(X)

    # -- point sets over a time window ---------------------------------------------

    def request_set(self, window):
        """-> the point set's metadata; starts building it unless it exists already"""
        if self.feature_fn is None:
            raise ValueError("start the explorer with feature_fn=controller_school_"
                             "features to build point sets over a time window")
        window = int(window)
        lo, hi = WINDOW_RANGE
        if not lo <= window <= hi:
            raise ValueError(f"window must be {lo} to {hi} iterations")
        with self._layout_lock:
            for record in self._sets:
                if record['window'] == window and record['status'] != 'error':
                    return _strip(record)
            record = {'id': len(self._sets), 'window': window, 'original': False,
                      'status': 'queued', 'points': None, 'columns': None,
                      'same_run': None, 'seconds': None, 'error': None, '_set': None}
            self._sets.append(record)
            meta = _strip(record)
        self._submit(('set', record))
        return meta

    def _build_set(self, record):
        t0 = time.time()
        with self._layout_lock:
            record['status'] = 'running'
        try:
            res = self._window_features(record['window'])
            X = np.asarray(res['X'], dtype=float)
            if len(X) < 3:
                raise ValueError(f"only {len(X)} points; the window is too long")
            Y = self._umap(X)
            res['embedding'] = self._align_to_first(Y, res['ids'], res['iterations'])
            ps = _PointSet(record['id'], res, window=record['window'],
                           seconds=round(time.time() - t0, 1))
            share = _same_run_share(X, res['ids'])
            with self._layout_lock:
                record.update(status='done', _set=ps, points=int(len(X)),
                              columns=int(X.shape[1]), same_run=share,
                              seconds=round(time.time() - t0, 1))
        except Exception as exc:   # report it on the page, keep the worker alive
            with self._layout_lock:
                record.update(status='error', error=f"{type(exc).__name__}: {exc}",
                              seconds=round(time.time() - t0, 1))

    def _window_features(self, window):
        """-> a result dict without an embedding, for a point set of this window"""
        out = self.feature_fn(self.data, window=window, window_stat='stack',
                              stride='window', **self.feature_kwargs)
        if isinstance(out, dict):
            return dict(out)
        X, ids, iterations, progress, pass_time, attack_dirs, blocks, features = out
        return {'X': X, 'ids': ids, 'iterations': iterations, 'progress': progress,
                'pass_time': pass_time, 'attack_dirs': list(attack_dirs),
                'blocks': blocks, 'features': features}

    def _align_to_first(self, Y, ids, iterations):
        """-> Y turned onto the first point set, pairing every point with the first
        set's point of the same run nearest in time, so a cluster stays roughly where
        it was when the window changes"""
        first = self._sets[0]['_set']
        ids, iterations = np.asarray(ids), np.asarray(iterations)
        pairs = []
        for run in np.unique(ids):
            mine = np.flatnonzero(ids == run)
            theirs = np.flatnonzero(first.ids == run)
            if not len(theirs):
                continue
            order = theirs[np.argsort(first.iterations[theirs])]
            t = first.iterations[order]
            k = np.clip(np.searchsorted(t, iterations[mine]), 1, len(t) - 1) if len(t) > 1 \
                else np.zeros(len(mine), dtype=int)
            if len(t) > 1:
                left = np.abs(t[k - 1] - iterations[mine]) <= np.abs(t[k] - iterations[mine])
                k = np.where(left, k - 1, k)
            pairs.append(np.column_stack([mine, order[k]]))
        if not pairs:
            return Y
        pairs = np.concatenate(pairs)
        Q, scale, mu_y, mu_r = _procrustes(Y[pairs[:, 0]], first.reference[pairs[:, 1]])
        return (Y - mu_y) @ Q * scale + mu_r

    # -- layouts over a subset of the metrics or directions -------------------------

    def _layouts_meta(self, set_id):
        ps = self._get_set(set_id)
        with self._layout_lock:
            return [_strip(layout) for layout in ps.layouts]

    def _layout_payload(self, set_id, layout_id):
        ps = self._get_set(set_id)
        with self._layout_lock:
            if not 0 <= layout_id < len(ps.layouts):
                raise KeyError(f"no layout {layout_id} in point set {set_id}")
            layout = ps.layouts[layout_id]
            out = _strip(layout)
            if layout['status'] == 'done':
                # points left out of the fit are NaN, and the page hides them
                out['x'], out['y'] = _f32(layout['x']), _f32(layout['y'])
        return json.dumps(out).encode()

    def request_layout(self, blocks, dirs=None, set_id=0, exclude=None):
        """-> the layout's metadata; starts a fit unless this one exists already

        blocks names the metrics to use. dirs names the attack directions whose points
        are laid out; None or every direction means all points. exclude keeps each
        point's neighbours out of its own run: 'run' for the whole run, a number of
        seconds for the moments of its run closer than that, None for no exclusion.
        """
        if exclude is not None and exclude != 'run':
            try:
                exclude = float(exclude)
            except (TypeError, ValueError):
                raise ValueError("exclude must be 'run', a number of seconds or None")
            if not 0 < exclude <= 600:
                raise ValueError("exclude must be more than 0 and at most 600 seconds")
        ps = self._get_set(set_id)
        if ps.X is None:
            raise ValueError("this result has no feature matrix 'X' and 'blocks', so "
                             "it cannot be laid out again")
        known = [name for name, _, _ in ps.choices]
        unknown = [b for b in blocks if b not in known]
        if unknown:
            raise ValueError(f"unknown metrics {unknown}; the result has {known}")
        wanted = [name for name in known if name in blocks]   # canonical order
        if not wanted:
            raise ValueError("pick at least one metric")
        if dirs is not None:
            unknown = [d for d in dirs if d not in ps.dir_names]
            if unknown:
                raise ValueError(f"unknown attack directions {unknown}; the result has "
                                 f"{ps.dir_names}")
            dirs = [d for d in ps.dir_names if d in dirs]
            if not dirs:
                raise ValueError("pick at least one attack direction")
            if len(dirs) == len(ps.dir_names):
                dirs = None
        with self._layout_lock:
            for layout in ps.layouts:
                if (layout['blocks'] == wanted and layout['dirs'] == dirs
                        and layout.get('exclude') == exclude
                        and layout['status'] != 'error'):
                    return _strip(layout)
            # in column order, so ticking everything gives back the original matrix
            columns = np.sort(np.concatenate([cols for name, cols, _ in ps.choices
                                              if name in wanted]))
            rows = (np.arange(len(ps.dirs)) if dirs is None
                    else np.flatnonzero(np.isin(ps.dirs, dirs)))
            layout = {'id': len(ps.layouts), 'blocks': wanted, 'dirs': dirs,
                      'exclude': exclude, 'same_run_2d': None,
                      'status': 'queued', 'original': False, 'points': int(len(rows)),
                      'columns': int(len(columns)),
                      'live': int(np.sum(ps.layout_matrix(rows, columns, dirs)
                                         .std(axis=0) > 0)),
                      'kept': None, 'seconds': None, 'error': None,
                      'x': None, 'y': None}
            ps.layouts.append(layout)
            meta = _strip(layout)
        self._submit(('layout', ps, layout, rows, columns))
        return meta

    # -- watershed clusters -----------------------------------------------------------

    def clusters(self, set_id=0, layout_id=0, k=None):
        """-> watershed regions of one layout's point density, as a JSON-ready dict

        Following Ehlman et al. 2025 (iScience), who follow MotionMapper: smooth the
        point density with a Gaussian, flood the inverse density from its minima, so
        every density peak becomes one region bordered by density valleys, and drop
        pixels below a density floor. k caps the number of regions: the narrowest
        smoothing that leaves at most k regions is used. With k None, the count that
        holds over the widest range of smoothing is used instead.
        """
        ps = self._get_set(set_id)
        with self._layout_lock:
            if not 0 <= layout_id < len(ps.layouts):
                raise KeyError(f"no layout {layout_id} in point set {set_id}")
            layout = ps.layouts[layout_id]
            if layout['status'] != 'done':
                raise ValueError(f"layout {layout_id} is not fitted yet")
            Y = np.column_stack([layout['x'], layout['y']])
        if k is not None:
            k = int(k)
            if not 2 <= k <= WSHED_MAX:
                raise ValueError(f"the number of regions must be 2 to {WSHED_MAX}")
        t0 = time.time()
        placed = np.isfinite(Y).all(axis=1)
        if placed.sum() < 10:
            raise ValueError("too few points in this layout to cluster")
        with self._wshed_lock:
            sweep = self._wshed.get((set_id, layout_id))
            if sweep is None:
                dmap = _DensityMap(Y[placed])
                # the floor can remove whole regions, so count the labels left
                counts = [int(np.count_nonzero(np.unique(dmap.regions(f * dmap.span)[0])))
                          for f in WSHED_SIGMAS]
                sweep = {'map': dmap, 'counts': counts,
                         'plateaus': _plateaus(WSHED_SIGMAS, counts)}
                self._wshed[(set_id, layout_id)] = sweep
        dmap, counts = sweep['map'], sweep['counts']
        if k is None:
            if not sweep['plateaus']:
                raise ValueError("the density has a single peak at every smoothing")
            _, lo, hi, _ = sweep['plateaus'][0]
            fraction = float(np.sqrt(lo * hi))
        else:
            fits = [f for f, c in zip(WSHED_SIGMAS, counts) if c <= k]
            if not fits:
                raise ValueError(f"even the widest smoothing leaves more than {k} regions")
            fraction = float(fits[0])
        grid, D = dmap.regions(fraction * dmap.span)

        # every point takes its pixel's region
        ix = np.clip(np.searchsorted(dmap.edges[0], Y[placed, 0]) - 1, 0, dmap.grid - 1)
        iy = np.clip(np.searchsorted(dmap.edges[1], Y[placed, 1]) - 1, 0, dmap.grid - 1)
        raw = grid[ix, iy]
        # number regions by size, 1 the largest; regions no point falls in are dropped
        ids, sizes = np.unique(raw[raw > 0], return_counts=True)
        order = ids[np.argsort(-sizes, kind='stable')]
        remap = np.zeros(grid.max() + 1, dtype=np.int32)
        remap[order] = np.arange(1, len(order) + 1)
        grid = remap[grid]
        n = len(order)
        labels = np.full(len(Y), -1, dtype=np.int32)    # -1: not in this layout
        labels[placed] = remap[raw]                     # 0: below the density floor
        peaks = []
        for r in range(1, n + 1):
            masked = np.where(grid == r, D, -1.0)
            i, j = np.unravel_index(np.argmax(masked), masked.shape)
            peaks.append([float((dmap.edges[0][i] + dmap.edges[0][i + 1]) / 2),
                          float((dmap.edges[1][j] + dmap.edges[1][j + 1]) / 2)])
        return {
            'set': set_id, 'layout': layout_id, 'k': k, 'regions': n,
            'sigma': fraction * dmap.span, 'sigma_fraction': fraction,
            'sizes': np.bincount(labels[labels > 0], minlength=n + 1)[1:].tolist(),
            'unassigned': int(np.sum(labels == 0)),
            'colours': _colour_regions(grid, n) if n else [],
            'peaks': peaks,
            'plateaus': [{'regions': c, 'from': lo, 'to': hi}
                         for c, lo, hi, _ in sweep['plateaus']],
            'sweep': [[float(f), int(c)] for f, c in zip(WSHED_SIGMAS, counts)],
            'labels': _i32(labels),
            # drawn only where the data is: the paper's floor leaves the whole map
            # covered, which would put borders out into empty space. The outline is
            # drawn at half the density of the sparsest 1% of points
            'borders': _f32(_boundaries(np.where(D < 0.5 * np.percentile(D[ix, iy], 1),
                                                 0, grid), dmap.edges)),
            'seconds': round(time.time() - t0, 2),
        }

    def _fit_layout(self, ps, layout, rows, columns):
        t0 = time.time()
        with self._layout_lock:
            layout['status'] = 'running'
        try:
            X = ps.layout_matrix(rows, columns, layout['dirs'])
            if len(rows) < 3:
                raise ValueError(f"only {len(rows)} points to lay out")
            if not np.any(X.std(axis=0) > 0):
                raise ValueError("none of these metrics varies, so there is "
                                 "nothing to lay out")
            reference = ps.reference[rows]
            Y = _align(self._umap(X, layout.get('exclude'), ps.ids[rows],
                                  ps.iterations[rows]), reference)
            kept = _neighbours_kept(Y, reference)
            same_run_2d = _same_run_share(Y, ps.ids[rows])
            full = np.full((len(ps.reference), 2), np.nan)
            full[rows] = Y
            with self._layout_lock:
                layout.update(status='done', x=full[:, 0], y=full[:, 1], kept=kept,
                              same_run_2d=same_run_2d,
                              seconds=round(time.time() - t0, 1))
        except Exception as exc:   # report it on the page, keep the worker alive
            with self._layout_lock:
                layout.update(status='error', error=f"{type(exc).__name__}: {exc}",
                              seconds=round(time.time() - t0, 1))


    # -- saved views ---------------------------------------------------------------

    def _save_path(self, name):
        """-> the path of a save file, refusing anything outside save_dir"""
        base = os.path.basename(str(name))
        if base != name or not base.endswith('.json'):
            raise ValueError(f"not a save file: {name!r}")
        return os.path.join(self.save_dir, base)

    def list_saves(self):
        """-> the saved views, newest first"""
        if not os.path.isdir(self.save_dir):
            return []
        out = []
        for name in os.listdir(self.save_dir):
            if not name.endswith('.json'):
                continue
            try:
                with open(os.path.join(self.save_dir, name), encoding='utf-8') as f:
                    doc = json.load(f)
                out.append({'file': name, 'name': doc.get('name', name),
                            'saved_at': doc.get('saved_at'), 'window': doc.get('window'),
                            'points': doc.get('n_points'),
                            'layout': _strip(doc.get('layout', {}))})
            except Exception:   # a damaged or foreign file is listed but marked
                out.append({'file': name, 'name': name, 'saved_at': None,
                            'error': 'unreadable'})
        return sorted(out, key=lambda d: d.get('saved_at') or '', reverse=True)

    def save_view(self, name, set_id, layout_id, view):
        """-> the new save's listing entry, after writing it to save_dir"""
        name = str(name or '').strip()[:120] or 'unnamed view'
        ps = self._get_set(set_id)
        with self._layout_lock:
            if not 0 <= layout_id < len(ps.layouts):
                raise KeyError(f"no layout {layout_id} in point set {set_id}")
            layout = ps.layouts[layout_id]
            if layout['status'] != 'done':
                raise ValueError(f"layout {layout_id} is {layout['status']}, not done")
            base = ps.layouts[0]
            doc = {
                'format': 1,
                'name': name,
                'saved_at': datetime.datetime.now().isoformat(timespec='seconds'),
                'title': self.title,
                'window': ps.window,
                'feature_fn': getattr(self.feature_fn, '__name__', None),
                'feature_kwargs': _jsonable(self.feature_kwargs),
                'umap': _jsonable(self._umap_kwargs),
                'n_points': int(len(ps.ids)),
                'features_hash': _fingerprint(ps.X),
                'points': {'run': _i32(ps.ids), 'it': _i32(ps.iterations)},
                'base': {'x': _f32(base['x']), 'y': _f32(base['y'])},
                'layout': dict(_jsonable(_strip(layout)), x=_f32(layout['x']),
                               y=_f32(layout['y'])),
                'view': _jsonable(view or {}),
            }
        os.makedirs(self.save_dir, exist_ok=True)
        slug = re.sub(r'[^A-Za-z0-9]+', '-', name).strip('-').lower()[:60] or 'view'
        stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        path = os.path.join(self.save_dir, f"{slug}_{stamp}.json")
        k = 1
        while os.path.exists(path):
            k += 1
            path = os.path.join(self.save_dir, f"{slug}_{stamp}_{k}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f)
        return {'file': os.path.basename(path), 'name': name, 'saved_at': doc['saved_at'],
                'window': ps.window, 'points': doc['n_points'], 'layout': _strip(layout)}

    def delete_save(self, file):
        if not self.allow_delete:
            raise ValueError("deleting saved views is turned off on this server")
        path = self._save_path(file)
        if not os.path.exists(path):
            raise KeyError(f"no saved view {file}")
        os.remove(path)
        return {'deleted': file}

    def load_view(self, file):
        """-> {'set', 'layout', 'view', 'warning'}: where the saved view now lives

        The saved points must be the same runs and iterations as a point set of this
        explorer. A windowed point set that this session has not built yet is rebuilt
        from feature_fn, with the saved embedding in place of a new UMAP fit, and the
        saved layout is added to it with its saved coordinates.
        """
        path = self._save_path(file)
        if not os.path.exists(path):
            raise KeyError(f"no saved view {file}")
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        ids = _decode(doc['points']['run'], '<i4')
        its = _decode(doc['points']['it'], '<i4')
        base = np.column_stack([_decode(doc['base']['x'], '<f4'),
                                _decode(doc['base']['y'], '<f4')]).astype(float)
        saved = doc['layout']
        coords = np.column_stack([_decode(saved['x'], '<f4'),
                                  _decode(saved['y'], '<f4')]).astype(float)
        window = doc.get('window')

        def same_points(ps):
            return (len(ps.ids) == len(ids) and np.array_equal(ps.ids, ids)
                    and np.array_equal(ps.iterations, its))

        with self._layout_lock:
            candidates = [r for r in self._sets if r['status'] == 'done'
                          and r['window'] == window and same_points(r['_set'])]
        if candidates:
            record = candidates[0]
        elif window is None:
            raise ValueError("this view was saved from a different embedding: its "
                             "points are not the runs and iterations the explorer was "
                             "started with")
        else:
            if self.feature_fn is None:
                raise ValueError("this view was saved from a point set over a "
                                 f"{window}-iteration window; start the explorer with "
                                 "feature_fn=controller_school_features to load it")
            res = self._window_features(window)
            res['embedding'] = base
            with self._layout_lock:
                record = {'id': len(self._sets), 'window': window, 'original': False,
                          'status': 'queued', 'points': None, 'columns': None,
                          'same_run': None, 'seconds': None, 'error': None, '_set': None}
                self._sets.append(record)
            ps = _PointSet(record['id'], res, window=window)
            if not same_points(ps):
                with self._layout_lock:
                    record.update(status='error', error='points differ from the save')
                raise ValueError("the features built now give other points than the "
                                 "saved ones; was the data or feature_fn changed?")
            X = np.asarray(res['X'], dtype=float)
            with self._layout_lock:
                record.update(status='done', _set=ps, points=int(len(X)),
                              columns=int(X.shape[1]), same_run=_same_run_share(X, ids))
        ps = record['_set']

        warning = None
        if doc.get('features_hash') and doc['features_hash'] != _fingerprint(ps.X):
            warning = ("the features differ from the ones the view was saved with, so "
                       "new layouts fitted from it will not match the saved one exactly")
        with self._layout_lock:
            for layout in ps.layouts:
                if (layout['status'] == 'done' and layout['x'] is not None
                        and np.allclose(np.column_stack([layout['x'], layout['y']]),
                                        coords, atol=1e-4, equal_nan=True)):
                    layout_id = layout['id']
                    break
            else:
                meta = {k: v for k, v in saved.items() if k not in ('x', 'y')}
                layout_id = len(ps.layouts)
                meta.update(id=layout_id, status='done', original=False, error=None,
                            saved=doc.get('name'), x=coords[:, 0], y=coords[:, 1])
                meta.setdefault('exclude', None)
                ps.layouts.append(meta)
        return {'set': record['id'], 'layout': layout_id, 'view': doc.get('view', {}),
                'name': doc.get('name'), 'warning': warning}

    # -- http ---------------------------------------------------------------------
    #
    #   GET  /api/sets                              point sets, and whether windows work
    #   POST /api/sets            {window}          build a point set
    #   GET  /api/sets/<s>/points                   one point set's embedding
    #   GET  /api/sets/<s>/layouts                  its layouts
    #   GET  /api/sets/<s>/layouts/<l>              one layout's coordinates
    #   POST /api/sets/<s>/layouts {blocks, dirs}   fit a layout
    #   GET  /api/run/<id>                          one run's positions
    #   GET  /api/points                            point set 0, as before
    #   GET  /api/saves                             saved views on disk
    #   POST /api/saves           {name, set, layout, view}   save one
    #   POST /api/saves/load      {file}            load one
    #   POST /api/saves/delete    {file}            delete one

    def _handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the notebook output clean
                pass

            def _send(self, body, content_type='application/json', status=200):
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)

            def _json(self, value):
                self._send(json.dumps(value).encode())

            def _error(self, status, message):
                self._send(json.dumps({'error': message}).encode(), status=status)

            def _body(self):
                length = int(self.headers.get('Content-Length') or 0)
                return json.loads(self.rfile.read(length) or b'{}')

            def do_GET(self):
                route = urlparse(self.path).path
                parts = route.strip('/').split('/')
                try:
                    if route in ('/', '/index.html'):
                        # read on every request, so edits show up on a reload
                        with open(HTML_PATH, 'rb') as f:
                            self._send(f.read(), 'text/html; charset=utf-8')
                    elif route == '/api/points':
                        self._send(app._points_payload(0))
                    elif route == '/api/sets':
                        self._json(app._sets_meta())
                    elif route == '/api/saves':
                        self._json({'dir': app.save_dir, 'saves': app.list_saves()})
                    elif parts[:2] == ['api', 'sets'] and len(parts) == 4 and parts[3] == 'points':
                        self._send(app._points_payload(int(parts[2])))
                    elif parts[:2] == ['api', 'sets'] and len(parts) == 4 and parts[3] == 'layouts':
                        self._json(app._layouts_meta(int(parts[2])))
                    elif parts[:2] == ['api', 'sets'] and len(parts) == 5 and parts[3] == 'layouts':
                        self._send(app._layout_payload(int(parts[2]), int(parts[4])))
                    elif parts[:2] == ['api', 'run'] and len(parts) == 3:
                        self._send(app._run_payload(int(parts[2])))
                    else:
                        self._error(404, f"no route {route}")
                except (KeyError, ValueError) as exc:
                    self._error(404, str(exc.args[0]) if exc.args else str(exc))
                except Exception as exc:  # report, never kill the server thread
                    self._error(500, f"{type(exc).__name__}: {exc}")

            def do_POST(self):
                route = urlparse(self.path).path
                parts = route.strip('/').split('/')
                try:
                    body = self._body()
                    if route == '/api/sets':
                        self._json(app.request_set(body.get('window')))
                    elif route == '/api/saves':
                        self._json(app.save_view(body.get('name'), int(body.get('set', 0)),
                                                 int(body.get('layout', 0)),
                                                 body.get('view')))
                    elif route == '/api/saves/load':
                        self._json(app.load_view(body.get('file')))
                    elif route == '/api/saves/delete':
                        self._json(app.delete_save(body.get('file')))
                    elif (parts[:2] == ['api', 'sets'] and len(parts) == 6
                          and parts[3] == 'layouts' and parts[5] == 'clusters'):
                        k = body.get('k')
                        self._json(app.clusters(int(parts[2]), int(parts[4]),
                                                None if k in (None, '') else int(k)))
                    elif parts[:2] == ['api', 'sets'] and len(parts) == 4 and parts[3] == 'layouts':
                        blocks, dirs = body.get('blocks'), body.get('dirs')
                        exclude = body.get('exclude')
                        if not isinstance(blocks, list):
                            raise ValueError("send {'blocks': [metric names]}")
                        if dirs is not None and not isinstance(dirs, list):
                            raise ValueError("'dirs' must be a list of attack directions")
                        self._json(app.request_layout(blocks, dirs, int(parts[2]), exclude))
                    else:
                        self._error(404, f"no route {route}")
                except (KeyError, ValueError, TypeError) as exc:
                    self._error(400, str(exc.args[0]) if exc.args else str(exc))
                except Exception as exc:
                    self._error(500, f"{type(exc).__name__}: {exc}")

        return Handler
