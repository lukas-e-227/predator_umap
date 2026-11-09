// Static copy of the UMAP explorer: answers the page's requests from the files that
// build_static.py exported, instead of from the Python server.
//
// The site holds several datasets, each exported to docs/api/<dataset>/. The page's
// URL picks one, as index.html?data=<dataset>; without it the first is shown.
// docs/api/datasets.json lists them.
//
// GET api/<path> reads docs/api/<dataset>/<path>.json. Loading a saved view reads the
// answer the server gave for it. Watershed clusters are computed here, in the browser, by the
// same steps as UmapExplorer.clusters in umap_explorer.py. Anything that needs UMAP,
// such as a new layout or a new time window, gets an error.
(() => {
  'use strict';
  const realFetch = window.fetch.bind(window);
  const reply = (obj, status = 200) => new Response(JSON.stringify(obj), {
    status, headers: { 'Content-Type': 'application/json' },
  });

  // the datasets, and the one this page shows
  let index = null;
  async function dataset() {
    if (!index) {
      index = realFetch('api/datasets.json').then(r => r.json()).then(list => {
        const asked = new URLSearchParams(window.location.search).get('data');
        const key = list.datasets.some(d => d.key === asked) ? asked : list.datasets[0].key;
        return { list: list.datasets, key };
      });
    }
    return index;
  }
  // a file of the shown dataset; api/sets/0 becomes api/<dataset>/sets/0
  const fileOf = async path => `api/${(await dataset()).key}/${path.slice(4)}`;

  window.fetch = async (input, opts = {}) => {
    const url = typeof input === 'string' ? input : input.url;
    if (!url.startsWith('api/')) return realFetch(input, opts);
    const path = url.split('?')[0];
    const method = (opts.method || 'GET').toUpperCase();
    if (method === 'GET') {
      if (path === 'api/points') return realFetch(await fileOf('api/sets/0/points.json'));
      if (path === 'api/sets') {
        // the point sets, and the datasets for the page's dataset menu
        const r = await realFetch(await fileOf('api/sets.json'));
        if (!r.ok) return r;
        const { list, key } = await dataset();
        return reply(Object.assign(await r.json(), { datasets: list, dataset: key }));
      }
      return realFetch(await fileOf(`${path}.json`));
    }
    const body = opts.body ? JSON.parse(opts.body) : {};
    if (path === 'api/saves/load') {
      return realFetch(await fileOf(`api/saves/load/${encodeURIComponent(body.file)}`));
    }
    const m = path.match(/^api\/sets\/(\d+)\/layouts\/(\d+)\/clusters$/);
    if (m) {
      try {
        return reply(await clusters(Number(m[1]), Number(m[2]), body.k ?? null));
      } catch (err) {
        return reply({ error: err.message }, 400);
      }
    }
    return reply({ error: 'This static copy cannot do that. It needs the Python explorer.' }, 400);
  };

  // -------------------------------------------------------------------------
  // watershed clusters, as in umap_explorer.py
  // -------------------------------------------------------------------------
  const GRID = 400;                       // density map pixels per side
  const FLOOR = 1e-5;                     // below this density a pixel is in no region
  const MAX_REGIONS = 60;
  const SIGMAS = Array.from({ length: 45 }, (_, i) => 0.003 * (0.15 / 0.003) ** (i / 44));

  const sweeps = new Map();               // "set:layout" -> the sweep over smoothing

  function decodeF32(b64) {
    const bin = atob(b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Float32Array(bytes.buffer);
  }

  class DensityMap {
    // a 2D histogram of the points, summing to 1, blurred by a Gaussian with peak 1
    constructor(xs, ys) {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < xs.length; i++) {
        x0 = Math.min(x0, xs[i]); x1 = Math.max(x1, xs[i]);
        y0 = Math.min(y0, ys[i]); y1 = Math.max(y1, ys[i]);
      }
      this.span = Math.max(x1 - x0, y1 - y0) || 1;
      const half = 0.55 * this.span;      // a 10% margin on every side
      this.lo = [(x0 + x1) / 2 - half, (y0 + y1) / 2 - half];
      this.pixel = 2 * half / GRID;
      this.edges = this.lo.map(l => Float64Array.from({ length: GRID + 1 }, (_, j) => l + j * this.pixel));
      this.H = new Float64Array(GRID * GRID);   // H[ix * GRID + iy]
      for (let i = 0; i < xs.length; i++) this.H[this.bin(0, xs[i]) * GRID + this.bin(1, ys[i])] += 1 / xs.length;
    }

    bin(axis, v) {
      // numpy.histogram2d's bin: the last edge belongs to the last bin
      return Math.min(GRID - 1, Math.max(0, Math.floor((v - this.lo[axis]) / this.pixel)));
    }

    cell(axis, v) {
      // searchsorted(edges, v) - 1, clipped, as clusters() assigns a point to a pixel
      return Math.min(GRID - 1, Math.max(0, Math.ceil((v - this.lo[axis]) / this.pixel) - 1));
    }

    density(sigma) {
      const s = sigma / this.pixel, r = Math.min(GRID - 1, Math.ceil(4 * s));
      const k = Float64Array.from({ length: 2 * r + 1 }, (_, j) => Math.exp(-0.5 * ((j - r) / s) ** 2));
      const tmp = new Float64Array(GRID * GRID), D = new Float64Array(GRID * GRID);
      // along y, then along x; zero outside the map
      for (let ix = 0; ix < GRID; ix++) {
        const row = ix * GRID;
        for (let iy = 0; iy < GRID; iy++) {
          const v = this.H[row + iy];
          if (!v) continue;
          const a = Math.max(0, iy - r), b = Math.min(GRID - 1, iy + r);
          for (let j = a; j <= b; j++) tmp[row + j] += v * k[j - iy + r];
        }
      }
      for (let ix = 0; ix < GRID; ix++) {
        const a = Math.max(0, ix - r), b = Math.min(GRID - 1, ix + r);
        for (let jx = a; jx <= b; jx++) {
          const w = k[jx - ix + r], src = ix * GRID, dst = jx * GRID;
          if (w < 1e-300) continue;
          for (let iy = 0; iy < GRID; iy++) {
            const v = tmp[src + iy];
            if (v) D[dst + iy] += v * w;
          }
        }
      }
      return D;
    }

    regions(sigma) {
      const D = this.density(sigma);
      return { labels: watershed(D), D };
    }
  }

  // neighbours of pixel p, 8-connected
  const OFFSETS = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];

  function watershed(D) {
    // flood the inverse density from its minima, which are the density's peaks. The
    // highest pixels go first; equal heights in the order they were reached
    const n = GRID * GRID, labels = new Int32Array(n);
    let next = 0;
    const isPeak = p => {
      if (!(D[p] > 0)) return false;
      const ix = (p / GRID) | 0, iy = p % GRID;
      for (const [dx, dy] of OFFSETS) {
        const jx = ix + dx, jy = iy + dy;
        if (jx < 0 || jy < 0 || jx >= GRID || jy >= GRID) continue;
        if (D[jx * GRID + jy] > D[p]) return false;
      }
      return true;
    };
    const peak = new Uint8Array(n);
    for (let p = 0; p < n; p++) peak[p] = isPeak(p) ? 1 : 0;
    // a flat top of several pixels is one peak
    const stack = [];
    for (let p = 0; p < n; p++) {
      if (!peak[p] || labels[p]) continue;
      labels[p] = ++next;
      stack.push(p);
      while (stack.length) {
        const q = stack.pop(), qx = (q / GRID) | 0, qy = q % GRID;
        for (const [dx, dy] of OFFSETS) {
          const jx = qx + dx, jy = qy + dy;
          if (jx < 0 || jy < 0 || jx >= GRID || jy >= GRID) continue;
          const t = jx * GRID + jy;
          if (peak[t] && !labels[t]) { labels[t] = next; stack.push(t); }
        }
      }
    }
    // a binary heap on (height, age), height being -D
    const hk = new Float64Array(n + 8), ha = new Float64Array(n + 8), hp = new Int32Array(n + 8);
    let size = 0, age = 0;
    const less = (i, j) => hk[i] < hk[j] || (hk[i] === hk[j] && ha[i] < ha[j]);
    const swap = (i, j) => {
      [hk[i], hk[j]] = [hk[j], hk[i]]; [ha[i], ha[j]] = [ha[j], ha[i]]; [hp[i], hp[j]] = [hp[j], hp[i]];
    };
    const push = p => {
      let i = size++;
      hk[i] = -D[p]; ha[i] = age++; hp[i] = p;
      while (i > 0) { const up = (i - 1) >> 1; if (!less(i, up)) break; swap(i, up); i = up; }
    };
    const pop = () => {
      const p = hp[0];
      size--;
      if (size > 0) {
        hk[0] = hk[size]; ha[0] = ha[size]; hp[0] = hp[size];
        let i = 0;
        while (true) {
          const l = 2 * i + 1, r = l + 1;
          let m = i;
          if (l < size && less(l, m)) m = l;
          if (r < size && less(r, m)) m = r;
          if (m === i) break;
          swap(i, m); i = m;
        }
      }
      return p;
    };
    for (let p = 0; p < n; p++) if (labels[p]) push(p);
    while (size) {
      const q = pop(), qx = (q / GRID) | 0, qy = q % GRID;
      for (const [dx, dy] of OFFSETS) {
        const jx = qx + dx, jy = qy + dy;
        if (jx < 0 || jy < 0 || jx >= GRID || jy >= GRID) continue;
        const t = jx * GRID + jy;
        if (!labels[t]) { labels[t] = labels[q]; push(t); }
      }
    }
    for (let p = 0; p < n; p++) if (D[p] < FLOOR) labels[p] = 0;
    return labels;
  }

  function countRegions(labels) {
    const seen = new Set();
    for (const v of labels) if (v) seen.add(v);
    return seen.size;
  }

  function plateaus(fractions, counts, top = 4) {
    // the counts that hold over the widest range of smoothing, widest first
    const step = Math.log(fractions[1] / fractions[0]), best = new Map();
    let start = 0;
    for (let i = 1; i <= counts.length; i++) {
      if (i === counts.length || counts[i] !== counts[start]) {
        if (counts[start] >= 2) {
          const lo = fractions[start], hi = fractions[i - 1], width = Math.log(hi / lo) + step;
          const run = [counts[start], lo, hi, width], old = best.get(counts[start]);
          if (!old || width > old[3]) best.set(counts[start], run);
        }
        start = i;
      }
    }
    return [...best.values()].sort((a, b) => b[3] - a[3]).slice(0, top);
  }

  function colourRegions(labels, n, colours = 8) {
    // greedy colouring: regions that touch differ where possible
    const adj = Array.from({ length: n + 1 }, () => new Set());
    for (let ix = 0; ix < GRID; ix++) {
      for (let iy = 0; iy < GRID; iy++) {
        const a = labels[ix * GRID + iy];
        if (!a) continue;
        for (const t of [ix + 1 < GRID ? (ix + 1) * GRID + iy : -1, iy + 1 < GRID ? ix * GRID + iy + 1 : -1]) {
          if (t < 0) continue;
          const b = labels[t];
          if (b && b !== a) { adj[a].add(b); adj[b].add(a); }
        }
      }
    }
    const order = Array.from({ length: n }, (_, i) => i + 1).sort((a, b) => adj[b].size - adj[a].size);
    const colour = new Map();
    for (const r of order) {
      const used = new Set([...adj[r]].filter(q => colour.has(q)).map(q => colour.get(q)));
      let pick = [...Array(colours).keys()].find(c => !used.has(c));
      if (pick === undefined) {
        const uses = c => [...adj[r]].filter(q => colour.get(q) === c).length;
        pick = [...Array(colours).keys()].reduce((a, c) => (uses(c) < uses(a) ? c : a), 0);
      }
      colour.set(r, pick);
    }
    return Array.from({ length: n }, (_, i) => colour.get(i + 1));
  }

  function boundaries(labels, edges) {
    // pixel edges between different labels, merged along each row and column
    const [ex, ey] = edges, out = [];
    for (let i = 0; i + 1 < GRID; i++) {       // between x bins i and i+1: vertical lines
      let a = -1;
      for (let j = 0; j <= GRID; j++) {
        const diff = j < GRID && labels[i * GRID + j] !== labels[(i + 1) * GRID + j];
        if (diff && a < 0) a = j;
        if (!diff && a >= 0) { out.push(ex[i + 1], ey[a], ex[i + 1], ey[j]); a = -1; }
      }
    }
    for (let j = 0; j + 1 < GRID; j++) {       // between y bins j and j+1: horizontal lines
      let a = -1;
      for (let i = 0; i <= GRID; i++) {
        const diff = i < GRID && labels[i * GRID + j] !== labels[i * GRID + j + 1];
        if (diff && a < 0) a = i;
        if (!diff && a >= 0) { out.push(ex[a], ey[j + 1], ex[i], ey[j + 1]); a = -1; }
      }
    }
    return Float32Array.from(out);
  }

  function percentile(values, q) {
    // numpy's default, linear between the two nearest ranks
    const v = Float64Array.from(values).sort(), pos = q / 100 * (v.length - 1);
    const lo = Math.floor(pos), hi = Math.min(v.length - 1, lo + 1);
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
  }

  const yieldFrame = () => new Promise(resolve => setTimeout(resolve, 0));

  async function clusters(setId, layoutId, k) {
    if (k != null && !(k >= 2 && k <= MAX_REGIONS)) throw new Error(`the number of regions must be 2 to ${MAX_REGIONS}`);
    const key = `${setId}:${layoutId}`;
    let sweep = sweeps.get(key);
    if (!sweep) {
      const r = await realFetch(await fileOf(`api/sets/${setId}/layouts/${layoutId}.json`));
      if (!r.ok) throw new Error(`no layout ${layoutId} in point set ${setId}`);
      const L = await r.json();
      const x = decodeF32(L.x), y = decodeF32(L.y), placed = [];
      for (let i = 0; i < x.length; i++) if (Number.isFinite(x[i]) && Number.isFinite(y[i])) placed.push(i);
      if (placed.length < 10) throw new Error('too few points in this layout to cluster');
      const map = new DensityMap(placed.map(i => x[i]), placed.map(i => y[i]));
      const counts = [];
      for (const f of SIGMAS) {
        counts.push(countRegions(map.regions(f * map.span).labels));
        await yieldFrame();                  // keep the page responsive meanwhile
      }
      sweep = { map, counts, plateaus: plateaus(SIGMAS, counts), x, y, placed };
      sweeps.set(key, sweep);
    }
    const { map, counts, x, y, placed } = sweep;
    let fraction;
    if (k == null) {
      if (!sweep.plateaus.length) throw new Error('the density has a single peak at every smoothing');
      const [, lo, hi] = sweep.plateaus[0];
      fraction = Math.sqrt(lo * hi);
    } else {
      const i = counts.findIndex(c => c <= k);
      if (i < 0) throw new Error(`even the widest smoothing leaves more than ${k} regions`);
      fraction = SIGMAS[i];
    }
    const { labels: grid, D } = map.regions(fraction * map.span);

    // every point takes its pixel's region
    const pix = placed.map(i => map.cell(0, x[i]) * GRID + map.cell(1, y[i]));
    const raw = pix.map(p => grid[p]);
    // number regions by size, 1 the largest; regions no point falls in are dropped
    const sizesRaw = new Map();
    for (const v of raw) if (v > 0) sizesRaw.set(v, (sizesRaw.get(v) || 0) + 1);
    const order = [...sizesRaw.keys()].sort((a, b) => sizesRaw.get(b) - sizesRaw.get(a) || a - b);
    let maxLabel = 0;
    for (const v of grid) if (v > maxLabel) maxLabel = v;
    const remap = new Int32Array(maxLabel + 1);
    order.forEach((v, i) => { remap[v] = i + 1; });
    const n = order.length;
    const regionGrid = grid.map(v => remap[v]);
    const labels = new Int32Array(x.length).fill(-1);    // -1: not in this layout
    placed.forEach((i, j) => { labels[i] = remap[raw[j]]; });
    const peaks = [];
    for (let r = 1; r <= n; r++) {
      let best = -1, bestD = -1;
      for (let p = 0; p < regionGrid.length; p++) if (regionGrid[p] === r && D[p] > bestD) { bestD = D[p]; best = p; }
      const ix = (best / GRID) | 0, iy = best % GRID;
      peaks.push([(map.edges[0][ix] + map.edges[0][ix + 1]) / 2, (map.edges[1][iy] + map.edges[1][iy + 1]) / 2]);
    }
    const sizes = new Array(n).fill(0);
    let unassigned = 0;
    for (const v of labels) { if (v > 0) sizes[v - 1]++; else if (v === 0) unassigned++; }
    // borders only where the data is: half the density of the sparsest 1% of points
    const cut = 0.5 * percentile(pix.map(p => D[p]), 1);
    const drawn = regionGrid.map((v, p) => (D[p] < cut ? 0 : v));
    return {
      set: setId, layout: layoutId, k, regions: n,
      sigma: fraction * map.span, sigma_fraction: fraction,
      sizes, unassigned,
      colours: n ? colourRegions(regionGrid, n) : [],
      peaks,
      plateaus: sweep.plateaus.map(([c, lo, hi]) => ({ regions: c, from: lo, to: hi })),
      sweep: SIGMAS.map((f, i) => [f, counts[i]]),
      // plain arrays, since JSON turns a typed array into an object; base64 from the server
      labels: Array.from(labels),
      borders: Array.from(boundaries(drawn, map.edges)),
    };
  }
})();
