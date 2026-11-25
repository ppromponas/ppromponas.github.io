/* <distillation-demo>: the interactive demo of "Online Scheduling for Entanglement Distillation:
 * Learning When to Wait" (P. Promponas, S. Tserkis, L. Tassiulas) as one self-contained custom element.
 *
 *   <script src="distillation-demo.js"></script>
 *   <distillation-demo></distillation-demo>
 *
 * The trained networks are in weights/ next to this file and are loaded when a setting is chosen.
 * Optional attributes: theme="auto" | "light" | "dark" (default auto), base="…/" (folder that holds weights/).
 * Built by distillation-demo-tools/build.py from src/engine.js and src/view.js. Do not edit by hand.
 */
(function (root) {
'use strict';
var DD = root.DistillDemo = root.DistillDemo || {};

/* ==========================================================================================
 * PART 1 OF 2: THE MODEL. No page, no drawing: this part also runs in Node, where
 * distillation-demo-tools/check_engine.mjs compares it with the Python simulator of the paper.
 *
 *   physics     Bell-diagonal pairs, memory decoherence, gate noise, the DEJMPS operation
 *   Link        the two nodes and their N memory pairs, one step at a time
 *   policies    Optimistic(n, alpha), Nested(n, alpha) with a fidelity threshold, and the learned policy
 *   Net         the trained network (encoder with self-attention, one score per action)
 *
 * It follows stochastic_source/simulator.py, paper_policies.py, nested_fidelity_threshold.py and
 * policy.py line by line; the names of the paper are used in the comments (slot = one step here).
 * ========================================================================================== */

/* ---------- physics ---------- */

/* A pair is Bell-diagonal: [A, B, C, D], A = fidelity. Depolarisation by a factor eta. */
function shrink(c, eta) {
  return [0.25 + eta * (c[0] - 0.25), 0.25 + eta * (c[1] - 0.25), 0.25 + eta * (c[2] - 0.25), 0.25 + eta * (c[3] - 0.25)];
}

/* DEJMPS on two pairs: the output if it succeeds, and the probability that it does (eqs. 2 and 3 of the paper). */
function dejmps(c1, c2) {
  var A = c1[0] * c2[0] + c1[3] * c2[3];
  var B = c1[0] * c2[3] + c1[3] * c2[0];
  var C = c1[2] * c2[2] + c1[1] * c2[1];
  var D = c1[2] * c2[1] + c1[1] * c2[2];
  var p = A + B + C + D;
  if (p <= 0) return { v: [0.25, 0.25, 0.25, 0.25], p: 0 };
  var inv = 1 / p;
  return { v: [A * inv, B * inv, C * inv, D * inv], p: p };
}

function joinTags(a, b) {                 /* the string of an operation: shorter input first, then alphabetical */
  var first = (a.length < b.length || (a.length === b.length && a <= b)) ? a : b;
  return '(' + first + (first === a ? b : a) + ')';
}

/* ---------- the link ---------- */

var FREE = 0, STOPPED = 1;

function emptySlot() {
  return { occ: false, A: 0, B: 0, C: 0, D: 0, r: 0, sigma: FREE, pc: [], piece: false, tag: '' };
}

/* cfg: N, F0, q, pg, lam (per-step memory factor), Fthr (deliver at or above it; Infinity = never by itself),
 *      rndGate(k), rndSrc(k): random numbers in [0, 1) for the outcome of the block in memory k and for
 *      the generation attempt of memory k. */
function Link(cfg) {
  this.N = cfg.N;
  this.F0 = cfg.F0;
  this.q = cfg.q;
  this.lam = cfg.lam;
  this.etaG = 1 - cfg.pg;
  this.Fthr = cfg.Fthr;
  this.herald = true;                     /* a memory emptied when the messages are read is not refilled in that step */
  this.rndGate = cfg.rndGate;
  this.rndSrc = cfg.rndSrc;
  this.nPair = this.N * (this.N - 1) / 2;
  this.nActions = this.nPair + 2 * this.N + 1;
  this.pairs = [];
  for (var i = 0; i < this.N; i++) for (var j = i + 1; j < this.N; j++) this.pairs.push([i, j]);
  this.slots = [];
  this.blocked = [];
  this.deliverNext = {};
  this.t = 0;
  for (var k = 0; k < this.N; k++) { this.slots.push(emptySlot()); this.blocked.push(false); }
}

Link.prototype.fnow = function (s) {
  return s.occ ? 0.25 + Math.pow(this.lam, s.r) * (s.A - 0.25) : 0;
};

Link.prototype.pBlock = function (s) {
  var p = 1;
  for (var i = 0; i < s.pc.length; i++) p *= s.pc[i];
  return p;
};

/* Start: empty memories, then the first generation attempts are read. Returns the memories that got a pair. */
Link.prototype.reset = function () {
  for (var k = 0; k < this.N; k++) { this.slots[k] = emptySlot(); this.blocked[k] = false; }
  this.deliverNext = {};
  this.t = 0;
  return this.admit([]);
};

/* What an operation on memories i and j would give, without applying it. */
Link.prototype.preview = function (i, j) {
  var si = this.slots[i], sj = this.slots[j];
  var ui = shrink([si.A, si.B, si.C, si.D], this.etaG * Math.pow(this.lam, si.r));
  var uj = shrink([sj.A, sj.B, sj.C, sj.D], this.etaG * Math.pow(this.lam, sj.r));
  var o = dejmps(ui, uj);
  var pb = o.p * this.pBlock(si) * this.pBlock(sj);
  return { v: o.v, p: o.p, pBlock: pb, Fif: 0.25 + this.lam * (o.v[0] - 0.25) };
};

Link.prototype.canPair = function (i, j) {
  var si = this.slots[i], sj = this.slots[j];
  return i !== j && si.occ && sj.occ && si.sigma === FREE && sj.sigma === FREE;
};

/* An operation on the pairs of memories i < j. The output stays in memory i; memory j is measured and free. */
Link.prototype.pair = function (i, j) {
  if (!this.canPair(i, j)) return null;
  var si = this.slots[i], sj = this.slots[j];
  var o = this.preview(i, j);
  var chain = si.pc.concat(sj.pc);
  chain.push(o.p);
  var tag = joinTags(si.tag, sj.tag);
  this.slots[j] = emptySlot();
  si.A = o.v[0]; si.B = o.v[1]; si.C = o.v[2]; si.D = o.v[3];
  si.r = 0; si.sigma = FREE; si.pc = chain; si.piece = false; si.tag = tag;
  return { keep: i, use: j, p: o.p, pBlock: o.pBlock, Fif: o.Fif };
};

/* The same operation with the output kept in memory `keep` (i or j). */
Link.prototype.pairKeep = function (i, j, keep) {
  var lo = Math.min(i, j), hi = Math.max(i, j);
  var res = this.pair(lo, hi);
  if (!res) return null;
  if (keep === hi) { var tmp = this.slots[lo]; this.slots[lo] = this.slots[hi]; this.slots[hi] = tmp; }
  res.keep = keep; res.use = keep === hi ? lo : hi;
  return res;
};

Link.prototype.canDiscard = function (i) {
  var s = this.slots[i];
  return s.occ && s.sigma === FREE && s.pc.length === 0;
};

Link.prototype.discard = function (i) {
  if (!this.canDiscard(i)) return false;
  this.slots[i] = emptySlot();
  return true;
};

Link.prototype.stop = function (i) {
  var s = this.slots[i];
  if (s.occ && s.sigma === FREE) s.sigma = STOPPED;
};

/* Blocks of this step whose pair is delivered when its outcomes arrive, whatever its fidelity (the baselines). */
Link.prototype.deliverIfVerified = function (ks) {
  for (var i = 0; i < ks.length; i++) this.deliverNext[ks[i]] = true;
};

/* End of the decisions: the outcomes and the heralds cross the link (one step), and are read.
 * Returns what happened: results of the blocks, deliveries, new pairs. */
Link.prototype.end = function () {
  var N = this.N, k, s, info = { results: [], delivered: [], arrivals: [], aged: [] };
  var blocked = [];
  for (k = 0; k < N; k++) blocked.push(false);
  for (k = 0; k < N; k++) {                              /* the outcomes of every block */
    s = this.slots[k];
    if (!s.occ) continue;
    if (s.pc.length) {
      var p = this.pBlock(s), ok = this.rndGate(k) < p;
      info.results.push({ k: k, ok: ok, p: p, ops: s.pc.length, tag: s.tag });
      if (ok) { s.r = 1; s.sigma = FREE; s.pc = []; s.piece = true; s.tag = '[' + s.tag + ']'; }
      else { this.slots[k] = emptySlot(); blocked[k] = this.herald; }
    } else {                                             /* a stored pair waits one more step */
      s.r += 1; s.sigma = FREE;
      info.aged.push(k);
    }
  }
  var named = this.deliverNext;
  this.deliverNext = {};
  for (k = 0; k < N; k++) {                              /* deliveries */
    s = this.slots[k];
    if (!s.occ) continue;
    var f = this.fnow(s);
    if (f >= this.Fthr || named[k]) {
      info.delivered.push({ k: k, F: f, tag: s.tag });
      this.slots[k] = emptySlot();
      blocked[k] = this.herald;
    }
  }
  this.blocked = blocked;
  this.t += 1;
  info.arrivals = this.admit(info.delivered);
  return info;
};

/* The heralds are read: every empty memory that made an attempt holds a new pair with probability q. */
Link.prototype.admit = function (delivered) {
  var w = (1 - this.F0) / 3, arrivals = [];
  for (var k = 0; k < this.N; k++) {
    var s = this.slots[k];
    if (s.occ || this.blocked[k]) continue;
    if (!(this.rndSrc(k) < this.q)) continue;
    s.occ = true; s.A = this.F0; s.B = w; s.C = w; s.D = w; s.r = 0; s.sigma = FREE; s.pc = []; s.piece = false; s.tag = 'o';
    arrivals.push(k);
    if (this.F0 >= this.Fthr) { delivered.push({ k: k, F: this.F0, tag: 'o' }); this.slots[k] = emptySlot(); }
  }
  for (k = 0; k < this.N; k++) this.blocked[k] = false;
  return arrivals;
};

/* The actions of the decision process, as numbers: pairs (i < j) in order, Stop(i), End, Discard(i). */
Link.prototype.decode = function (a) {
  if (a < this.nPair) return { kind: 'PAIR', i: this.pairs[a][0], j: this.pairs[a][1] };
  if (a < this.nPair + this.N) return { kind: 'STOP', i: a - this.nPair };
  if (a === this.nPair + this.N) return { kind: 'END' };
  return { kind: 'DISCARD', i: a - this.nPair - this.N - 1 };
};

Link.prototype.mask = function () {
  var m = new Uint8Array(this.nActions), N = this.N, k;
  for (var a = 0; a < this.nPair; a++) if (this.canPair(this.pairs[a][0], this.pairs[a][1])) m[a] = 1;
  for (k = 0; k < N; k++) if (this.slots[k].occ && this.slots[k].sigma === FREE) m[this.nPair + k] = 1;
  m[this.nPair + N] = 1;
  for (k = 0; k < N; k++) if (this.canDiscard(k)) m[this.nPair + N + 1 + k] = 1;
  return m;
};

/* What the network sees: 14 numbers per memory pair. */
Link.prototype.features = function () {
  var N = this.N, f = new Float32Array(N * 14);
  for (var k = 0; k < N; k++) {
    var s = this.slots[k], o = k * 14;
    if (s.occ) {
      f[o] = 1;
      f[o + 1] = this.fnow(s);
      f[o + 2] = s.B; f[o + 3] = s.C; f[o + 4] = s.D;
      f[o + 5] = Math.pow(this.lam, s.r);
      f[o + 6] = s.sigma === STOPPED ? 1 : 0;
      f[o + 7] = Math.min(s.pc.length, 8) / 8;
      f[o + 8] = this.pBlock(s);
    }
    f[o + 9] = this.Fthr;
    f[o + 10] = this.lam;
    f[o + 11] = this.etaG;
    f[o + 12] = 1 / N;
    f[o + 13] = this.F0;
  }
  return f;
};

/* ---------- the two baselines ---------- */

/* Fidelity of a pair of level k if nothing in its tree waited (the yardstick of the fidelity cutoff). */
function freshFidelities(link, d) {
  var v = [link.F0, (1 - link.F0) / 3, (1 - link.F0) / 3, (1 - link.F0) / 3], out = [link.F0];
  for (var k = 1; k <= d; k++) {
    var u = shrink(v, link.etaG * Math.pow(link.lam, k === 1 ? 0 : 1));
    v = dejmps(u, u).v;
    out.push(0.25 + link.lam * (v[0] - 0.25));
  }
  return out;
}

/* kind 'optimistic': Optimistic(n, alpha). kind 'nested': Nested(n, alpha), delivering a pair of the top level
 * only at or above Fthr and distilling it further otherwise (Section V-C of the paper). */
function Baseline(link, kind, n, alpha, Fthr) {
  this.link = link;
  this.kind = kind;
  this.n = n;
  this.d = Math.round(Math.log(n) / Math.LN2);
  this.alpha = alpha;
  this.Fthr = Fthr;
  this.fresh = freshFidelities(link, this.d);
  this.level = {};                          /* stored pair -> level */
  this.pending = {};                        /* memory -> [level of the output, A of the output, p] */
}

Baseline.prototype.keys = function () {
  var ks = [];
  for (var k in this.level) ks.push(+k);
  return ks;
};

Baseline.prototype.reconcile = function () {
  var slots = this.link.slots, k;
  for (k in this.pending) {
    var s = slots[k];
    if (s.occ && s.piece) this.level[k] = this.pending[k][0];
  }
  this.pending = {};
  for (k = 0; k < slots.length; k++) if (slots[k].occ && !(k in this.level)) this.level[k] = 0;
};

/* The decisions of one step. Returns the operations, the discards and the blocks named for delivery. */
Baseline.prototype.decide = function () {
  var self = this, link = this.link, slots = link.slots, out = { ops: [], discards: [], named: [] };
  this.reconcile();
  function gate(a, b) {
    var r = link.pairKeep(a, b, a);
    out.ops.push(r);
    return r;
  }
  function discard(k) {
    link.discard(k);
    delete self.level[k];
    out.discards.push(k);
  }
  function byAge(ks) { return ks.sort(function (x, y) { return (slots[y].r - slots[x].r) || (x - y); }); }
  var ks, i;
  if (this.kind === 'optimistic') {
    ks = this.keys().sort(function (x, y) { return x - y; });
    for (i = 0; i < ks.length; i++) if (link.fnow(slots[ks[i]]) < this.alpha * this.fresh[0]) discard(ks[i]);
    while (this.keys().length >= this.n) {
      var leaves = byAge(this.keys()).slice(0, this.n), nodes = leaves.slice();
      while (nodes.length > 1) {
        var next = [];
        for (i = 0; i < nodes.length; i += 2) { gate(nodes[i], nodes[i + 1]); next.push(nodes[i]); }
        nodes = next;
      }
      var root = nodes[0], sr = slots[root];
      for (i = 0; i < leaves.length; i++) delete this.level[leaves[i]];
      this.pending[root] = [this.d, sr.A, link.pBlock(sr)];
      out.named.push(root);
    }
    return out;
  }
  /* nested: the cutoff first (highest level first), then the pairs of every level two by two, oldest first */
  ks = this.keys().sort(function (x, y) { return (self.level[y] - self.level[x]) || (x - y); });
  for (i = 0; i < ks.length; i++) {
    var k = ks[i], f = link.fnow(slots[k]), L = this.level[k];
    while (L >= 1 && f < this.alpha * this.fresh[L - 1]) L -= 1;
    if (L === 0 && f < this.alpha * this.fresh[0]) discard(k);
    else this.level[k] = L;
  }
  var top = [];
  for (var lvl = 0; lvl < this.d; lvl++) {
    var pool = byAge(this.keys().filter(function (x) { return self.level[x] === lvl; }));
    for (i = 0; i + 1 < pool.length; i += 2) {
      var a = pool[i], b = pool[i + 1], r = gate(a, b);
      delete this.level[a];
      delete this.level[b];
      this.pending[a] = [lvl + 1, slots[a].A, r.p];
      if (lvl + 1 === this.d) top.push(a);
    }
  }
  for (i = 0; i < top.length; i++) {
    var pd = this.pending[top[i]];
    if (0.25 + link.lam * (pd[1] - 0.25) >= this.Fthr) out.named.push(top[i]);
    else this.pending[top[i]] = [this.d - 1, pd[1], pd[2]];
  }
  return out;
};

/* ---------- the learned policy ---------- */

function erf(x) {
  var ax = Math.abs(x), r;
  if (ax < 2.5) {
    var sum = ax, term = ax, x2 = ax * ax;
    for (var n = 1; n < 90; n++) {
      term *= -x2 / n;
      var add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
    }
    r = 1.1283791670955126 * sum;
  } else {
    var t = 0;
    for (var k = 60; k >= 1; k--) t = (k / 2) / (ax + t);
    r = 1 - Math.exp(-ax * ax) / ((ax + t) * 1.7724538509055159);
  }
  return x < 0 ? -r : r;
}

function gelu(x) { return 0.5 * x * (1 + erf(x * 0.7071067811865476)); }

function decodeBase64(b64) {
  var bin;
  if (typeof atob === 'function') bin = atob(b64);
  else bin = Buffer.from(b64, 'base64').toString('binary');
  var bytes = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

/* payload: one of the files in weights/ (shapes of the tensors and their float32 values). */
function Net(payload) {
  var net = payload.net, all = decodeBase64(payload.data), off = 0;
  this.d = net.d_emb;
  this.L = net.n_layers;
  this.H = net.n_heads;
  this.ffn = net.ffn_hidden;
  this.dSlot = net.d_slot;
  this.w = {};
  for (var i = 0; i < net.shapes.length; i++) {
    var size = 1, sh = net.shapes[i][1];
    for (var j = 0; j < sh.length; j++) size *= sh[j];
    this.w[net.shapes[i][0]] = all.subarray(off, off + size);
    off += size;
  }
  if (off !== all.length) throw new Error('weights: wrong size');
}

function linear(x, n, din, W, b, dout) {            /* rows of x times W transposed, plus b */
  var y = new Float64Array(n * dout);
  for (var r = 0; r < n; r++) {
    var xr = r * din, yr = r * dout;
    for (var o = 0; o < dout; o++) {
      var s = b[o], wo = o * din;
      for (var i = 0; i < din; i++) s += W[wo + i] * x[xr + i];
      y[yr + o] = s;
    }
  }
  return y;
}

function layerNorm(x, n, d, g, b) {
  for (var r = 0; r < n; r++) {
    var o = r * d, m = 0, v = 0, i;
    for (i = 0; i < d; i++) m += x[o + i];
    m /= d;
    for (i = 0; i < d; i++) { var e = x[o + i] - m; v += e * e; }
    var inv = 1 / Math.sqrt(v / d + 1e-5);
    for (i = 0; i < d; i++) x[o + i] = (x[o + i] - m) * inv * g[i] + b[i];
  }
  return x;
}

/* The embedding of every memory pair: the same small network for each, then self-attention layers. */
Net.prototype.embed = function (feats, n) {
  var w = this.w, d = this.d, H = this.H, dh = d / H, i, j, r, c, h;
  var x = linear(feats, n, this.dSlot, w['slot_encoder.net.0.weight'], w['slot_encoder.net.0.bias'], d);
  for (i = 0; i < x.length; i++) x[i] = gelu(x[i]);
  x = linear(x, n, d, w['slot_encoder.net.2.weight'], w['slot_encoder.net.2.bias'], d);
  layerNorm(x, n, d, w['slot_encoder.net.3.weight'], w['slot_encoder.net.3.bias']);
  var scale = 1 / Math.sqrt(dh), att = new Float64Array(n);
  for (var l = 0; l < this.L; l++) {
    var p = 'gat.layers.' + l + '.';
    var qkv = linear(x, n, d, w[p + 'qkv.weight'], w[p + 'qkv.bias'], 3 * d);
    var z = new Float64Array(n * d);
    for (h = 0; h < H; h++) {
      for (r = 0; r < n; r++) {
        var qo = r * 3 * d + h * dh, mx = -Infinity, sum = 0;
        for (c = 0; c < n; c++) {
          var ko = c * 3 * d + d + h * dh, s = 0;
          for (i = 0; i < dh; i++) s += qkv[qo + i] * qkv[ko + i];
          s *= scale;
          att[c] = s;
          if (s > mx) mx = s;
        }
        for (c = 0; c < n; c++) { att[c] = Math.exp(att[c] - mx); sum += att[c]; }
        for (c = 0; c < n; c++) {
          var a = att[c] / sum, vo = c * 3 * d + 2 * d + h * dh, zo = r * d + h * dh;
          for (i = 0; i < dh; i++) z[zo + i] += a * qkv[vo + i];
        }
      }
    }
    z = linear(z, n, d, w[p + 'proj.weight'], w[p + 'proj.bias'], d);
    for (i = 0; i < x.length; i++) x[i] += z[i];
    layerNorm(x, n, d, w[p + 'norm1.weight'], w[p + 'norm1.bias']);
    var ff = linear(x, n, d, w[p + 'ffn.0.weight'], w[p + 'ffn.0.bias'], this.ffn);
    for (i = 0; i < ff.length; i++) ff[i] = gelu(ff[i]);
    ff = linear(ff, n, this.ffn, w[p + 'ffn.2.weight'], w[p + 'ffn.2.bias'], d);
    for (i = 0; i < x.length; i++) x[i] += ff[i];
    layerNorm(x, n, d, w[p + 'norm2.weight'], w[p + 'norm2.bias']);
  }
  return x;
};

/* One score per action, in the order of Link.decode. Actions that are not possible get -Infinity. */
Net.prototype.scores = function (feats, n, mask) {
  var w = this.w, d = this.d, h = this.embed(feats, n), i, j, e, k;
  var W = w['policy_head.W_pair'];
  var first = linear(h, n, d, w['policy_head.first_pick.weight'], w['policy_head.first_pick.bias'], 1);
  var hW = new Float64Array(n * d);                 /* row i: h_i times W */
  for (i = 0; i < n; i++) for (k = 0; k < d; k++) {
    var hv = h[i * d + k], wo = k * d, ro = i * d;
    for (e = 0; e < d; e++) hW[ro + e] += hv * W[wo + e];
  }
  var M = new Float64Array(n * n);
  for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
    var s = 0;
    for (e = 0; e < d; e++) s += hW[i * d + e] * h[j * d + e];
    M[i * n + j] = s;
  }
  var nPair = n * (n - 1) / 2, out = new Float64Array(nPair + 2 * n + 1), a = 0;
  for (i = 0; i < n; i++) for (j = i + 1; j < n; j++) {
    var u = first[i] + M[i * n + j], v = first[j] + M[j * n + i], m = Math.max(u, v);
    out[a++] = m + Math.log(Math.exp(u - m) + Math.exp(v - m));
  }
  var stop = linear(h, n, d, w['policy_head.w_stop.weight'], w['policy_head.w_stop.bias'], 1);
  for (i = 0; i < n; i++) out[a++] = stop[i];
  var mean = new Float64Array(d);
  for (i = 0; i < n; i++) for (e = 0; e < d; e++) mean[e] += h[i * d + e] / n;
  out[a++] = linear(mean, 1, d, w['policy_head.w_end.weight'], w['policy_head.w_end.bias'], 1)[0];
  var del = linear(h, n, d, w['policy_head.w_delete.weight'], w['policy_head.w_delete.bias'], 1);
  for (i = 0; i < n; i++) out[a++] = del[i];
  if (mask) for (i = 0; i < out.length; i++) if (!mask[i]) out[i] = -Infinity;
  return out;
};

/* The learned policy: in every state, the possible action with the highest score, until it ends the step. */
function Learned(link, net) {
  this.link = link;
  this.net = net;
}

Learned.prototype.choose = function () {
  var link = this.link, sc = this.net.scores(link.features(), link.N, link.mask()), best = 0;
  for (var i = 1; i < sc.length; i++) if (sc[i] > sc[best]) best = i;
  return best;
};

Learned.prototype.decide = function () {
  var link = this.link, out = { ops: [], discards: [], named: [], actions: [] };
  for (;;) {
    var a = this.choose(), act = link.decode(a);
    out.actions.push(a);
    if (act.kind === 'END') return out;
    if (act.kind === 'PAIR') out.ops.push(link.pair(act.i, act.j));
    else if (act.kind === 'DISCARD') { link.discard(act.i); out.discards.push(act.i); }
    else link.stop(act.i);
  }
};

/* ---------- trees from the strings of the paper ---------- */

/* 'o' is an elementary pair, '(xy)' an operation, '[x]' x used only after its outcomes were known.
 * Returns {leaf: true} or {a, b, waited: true|false}. */
function parseTag(tag) {
  var i = 0;
  function node() {
    var c = tag.charAt(i);
    if (c === 'o') { i++; return { leaf: true }; }
    if (c === '[') { i++; var x = node(); i++; x.waited = true; return x; }
    i++;                                            /* '(' */
    var a = node(), b = node();
    i++;                                            /* ')' */
    return { a: a, b: b, waited: false };
  }
  return node();
}

function stripOuter(tag) { return tag.charAt(0) === '[' ? tag.slice(1, -1) : tag; }

DD.model = { shrink: shrink, dejmps: dejmps, Link: Link, Baseline: Baseline, Net: Net, Learned: Learned,
             parseTag: parseTag, stripOuter: stripOuter, freshFidelities: freshFidelities, erf: erf };

/* ==========================================================================================
 * PART 2 OF 2: THE PAGE ELEMENT <distillation-demo>.
 *
 * Two halves run on the same generation luck: on the left a baseline of the paper (Nested, Optimistic) or
 * the visitor, on the right the learned policy. Each half draws
 *   the two nodes, with the memories as circles inside them, and the link between them;
 *   the circuit of the link, step after step, in a window that scrolls back in time;
 *   the trees now in memory, and the trees delivered so far with their shares.
 * Words on screen: step (the time slot of the paper), Send outcomes, memory, pair, fidelity.
 * ========================================================================================== */
if (root.customElements && root.document) (function () {
  var M = DD.model, doc = root.document;
  var here = doc.currentScript;
  var BASE = here && here.src ? here.src.replace(/[^\/]*$/, '') : '';

  /* ---------- what can be chosen ---------- */
  var PRESETS = [
    { id: 'f090', F0: 0.90, q: 0.5, T: 200, Fthr: 0.95 },
    { id: 'q1_095', F0: 0.85, q: 1, T: 200, Fthr: 0.95 },
    { id: 'q1_097', F0: 0.85, q: 1, T: 200, Fthr: 0.97 }
  ];
  /* Next to the long-run throughput of the three policies, the same text in every setting. It is the last sentence
   * of the abstract of the paper, with the two baselines named first; "for the outcomes of some operations but not
   * of others" is from the conclusion. */
  var COMPARE = 'Nested always waits for the outcomes of an operation before using its output, and Optimistic never waits. ' +
    'Which of the two is better depends on the setting. ' +
    'The learned policy delivers as many pairs as the better one when one of these two extremes is the best choice, and more otherwise. ' +
    'In that case, it waits for the outcomes of some operations but not of others.';
  var MODES = [
    { id: 'nested', name: 'Nested', sub: 'waits for every outcome' },
    { id: 'optimistic', name: 'Optimistic', sub: 'never waits for an outcome' },
    { id: 'you', name: 'You', sub: 'you decide' }
  ];
  var SPEEDS = [{ id: 1, name: '1×' }, { id: 2, name: '2×' }, { id: 4, name: '4×' }, { id: 0, name: 'Max' }];
  /* how long each part of a step is shown at 1×, in milliseconds */
  var PACE = { first: 1400, arrive: 2000, op: 850, after: 1400, idle: 900, travel: 1700 };
  var PAPER = 'https://arxiv.org/abs/2611.28373';

  /* ---------- small tools ---------- */
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function r2(v) { return String(Math.round(v * 100) / 100); }
  function f4(v) { return v.toFixed(4); }
  function pct(v) { return Math.round(v * 100) + '%'; }
  function plural(n, one, many) { return n + ' ' + (n === 1 ? one : (many || one + 's')); }
  function leavesOf(tag) { var n = 0; for (var i = 0; i < tag.length; i++) if (tag.charAt(i) === 'o') n++; return n; }

  /* The same luck for both halves: a random number that depends only on the step, the memory and its use. */
  function luck(seed, t, k, use) {
    var h = (seed ^ Math.imul(t + 1, 0x9E3779B1) ^ Math.imul(k + 1, 0x85EBCA77) ^ Math.imul(use + 1, 0xC2B2AE3D)) >>> 0;
    h ^= h >>> 16; h = Math.imul(h, 0x7FEB352D); h ^= h >>> 15; h = Math.imul(h, 0x846CA68B); h ^= h >>> 16;
    return (h >>> 0) / 4294967296;
  }

  var pendingLoads = {};
  function loadWeights(id, base) {
    var store = root.__distillDemoWeights || (root.__distillDemoWeights = {});
    if (store[id]) return Promise.resolve(store[id]);
    if (!pendingLoads[id]) pendingLoads[id] = new Promise(function (ok, no) {
      var s = doc.createElement('script');
      s.src = base + 'weights/' + id + '.js';
      s.async = true;
      s.onload = function () { if (store[id]) ok(store[id]); else no(new Error('weights')); };
      s.onerror = function () { delete pendingLoads[id]; no(new Error('weights')); };
      doc.head.appendChild(s);
    });
    return pendingLoads[id];
  }

  /* ---------- drawing a tree from its string ---------- */
  function strHTML(tag) {
    return esc(tag).replace(/[\[\]]/g, function (c) { return '<i>' + c + '</i>'; });
  }

  /* delivered = true: every operation is confirmed, and the last one is drawn as waited for, since a pair is
   * delivered only when the outcomes of its last operation have arrived. Otherwise the operations outside
   * every bracket are those of the block being built, drawn hollow. */
  function treeSVG(tag, delivered, w, h) {
    var t = M.parseTag(tag);
    function count(n) { return n.leaf ? 1 : count(n.a) + count(n.b); }
    function height(n) { return n.leaf ? 0 : 1 + Math.max(height(n.a), height(n.b)); }
    var nl = count(t), ht = height(t);
    if (!ht) return '<svg viewBox="0 0 ' + w + ' ' + h + '" aria-hidden="true"><circle class="leaf" cx="' + w / 2 + '" cy="' + h / 2 + '" r="3.4"/></svg>';
    var dy = Math.min(9.5, (h - 10) / (nl - 1)), dx = Math.min(17, (w - 12) / ht);
    var ox = (w - ht * dx) / 2, oy = (h - (nl - 1) * dy) / 2, edges = '', nodes = '';
    function place(n, y0, inside, top) {
      if (n.leaf) { var py = oy + y0 * dy; nodes += '<circle class="leaf" cx="' + r2(ox) + '" cy="' + r2(py) + '" r="2.9"/>'; return { x: ox, y: py, n: 1 }; }
      var conf = inside || n.waited || delivered;
      var A = place(n.a, y0, conf, false), B = place(n.b, y0 + A.n, conf, false);
      var X = ox + height(n) * dx, Y = (A.y + B.y) / 2;
      [A, B].forEach(function (c) {
        edges += '<line class="edge' + (conf ? '' : ' edge-p') + '" x1="' + r2(c.x) + '" y1="' + r2(c.y) + '" x2="' + r2(X) + '" y2="' + r2(Y) + '"/>';
      });
      if (n.waited || (delivered && top)) nodes += '<path class="op-w" d="M' + r2(X - 4.3) + ' ' + r2(Y) + 'L' + r2(X) + ' ' + r2(Y - 4.3) + 'L' + r2(X + 4.3) + ' ' + r2(Y) + 'L' + r2(X) + ' ' + r2(Y + 4.3) + 'Z"/>';
      else nodes += '<circle class="' + (conf ? 'op-n' : 'op-p') + '" cx="' + r2(X) + '" cy="' + r2(Y) + '" r="3.3"/>';
      return { x: X, y: Y, n: A.n + B.n };
    }
    place(t, 0, false, true);
    return '<svg viewBox="0 0 ' + w + ' ' + h + '" aria-hidden="true">' + edges + nodes + '</svg>';
  }

  /* ---------- one half of the demo: a link and whoever schedules it ---------- */
  function snap(link) {
    return link.slots.map(function (s) {
      return { occ: s.occ, F: link.fnow(s), tag: s.tag, pend: s.pc.length > 0, ops: s.pc.length,
               Fif: 0.25 + link.lam * (s.A - 0.25), p: link.pBlock(s) };
    });
  }
  function copySlots(link) {
    return link.slots.map(function (s) {
      return { occ: s.occ, A: s.A, B: s.B, C: s.C, D: s.D, r: s.r, sigma: s.sigma, pc: s.pc.slice(), piece: s.piece, tag: s.tag };
    });
  }

  function Player(side, mode, payload, net, seed) {
    var S = payload.setting, self = this, auto = mode !== 'you';
    this.side = side;
    this.mode = mode;
    this.thr = S.F_thr;
    this.N = S.N;
    this.link = new M.Link({
      N: S.N, F0: S.F0, q: S.q, pg: S.p_g, lam: S.lam,
      Fthr: (mode === 'nested' || mode === 'optimistic') ? Infinity : S.F_thr,
      rndGate: function (k) { return luck(seed, self.link.t, k, 2); },
      rndSrc: function (k) { return luck(seed, self.link.t, k, 1); }
    });
    if (mode === 'learned') this.pol = new M.Learned(this.link, net);
    else if (auto) this.pol = new M.Baseline(this.link, mode, payload.paper[mode].n, payload.paper[mode].alpha, S.F_thr);
    else this.pol = null;
    this.delivered = 0;                 /* pairs delivered at or above the required fidelity */
    this.low = 0;                       /* pairs delivered below it (the two rules only) */
    this.steps = 0;
    this.tally = {};
    this.hist = [];                     /* the circuit, one record per finished step */
    this.ops = [];
    this.shown = 0;
    this.named = [];
    this.undo = [];
    this.sel = null;
    this.hover = null;
    this.phase = 'arrive';
    /* The memories are empty at the start, as in the paper: the first step only makes generation attempts. */
    this.info = { results: [], delivered: [], arrivals: [], aged: [], first: true };
    this.before = snap(this.link);
    this.after = this.before;
    this.rec = this.openRecord([]);
  }

  /* The number of the current step, counted from 1. While the outcomes travel it is still the step that sent them. */
  Player.prototype.stepNo = function () { return this.link.t + (this.phase === 'travel' ? 0 : 1); };

  /* The deliveries read at the start of a step are counted when that step begins, not while the outcomes travel. */
  Player.prototype.reveal = function () {
    var self = this;
    this.info.delivered.forEach(function (d) {
      if (d.F >= self.thr) {
        self.delivered++;
        var tag = M.stripOuter(d.tag);
        self.tally[tag] = (self.tally[tag] || 0) + 1;
      } else self.low++;
    });
  };

  Player.prototype.openRecord = function (arrivals) {
    var rec = { t: this.link.t, start: [], ops: [], discards: [], sent: false, end: [], outcome: [] };
    for (var k = 0; k < this.N; k++) {
      rec.start.push(this.link.slots[k].occ ? (arrivals.indexOf(k) >= 0 ? 'new' : 'carry') : null);
      rec.end.push(null);
      rec.outcome.push(null);
    }
    return rec;
  };

  function columnFor(rec, keep, use) {            /* an operation is drawn after those it depends on or would cross */
    var lo = Math.min(keep, use), hi = Math.max(keep, use), col = 0;
    rec.ops.forEach(function (o) {
      var l = Math.min(o.keep, o.use), h = Math.max(o.keep, o.use);
      if (!(h < lo || l > hi)) col = Math.max(col, o.col + 1);
    });
    return col;
  }

  Player.prototype.addOp = function (res) {
    res.col = columnFor(this.rec, res.keep, res.use);
    this.rec.ops.push(res);
    this.ops.push(res);
  };

  Player.prototype.decideAuto = function () {
    var out = this.pol.decide(), self = this;
    out.ops.forEach(function (o) { self.addOp(o); });
    this.rec.discards = out.discards.slice();
    this.named = out.named;
    this.shown = 0;
  };

  Player.prototype.userOp = function (keep, use) {
    if (!this.link.canPair(keep, use)) return false;
    this.undo.push({ slots: copySlots(this.link), nOps: this.ops.length, discards: this.rec.discards.slice() });
    this.addOp(this.link.pairKeep(keep, use, keep));
    this.shown = this.ops.length;
    return true;
  };

  Player.prototype.userDiscard = function (k) {
    if (!this.link.canDiscard(k)) return false;
    this.undo.push({ slots: copySlots(this.link), nOps: this.ops.length, discards: this.rec.discards.slice() });
    this.link.discard(k);
    this.rec.discards.push(k);
    return true;
  };

  Player.prototype.userUndo = function () {
    var u = this.undo.pop();
    if (!u) return;
    this.link.slots = u.slots;
    this.ops.length = u.nOps;
    this.rec.ops.length = u.nOps;
    this.rec.discards = u.discards;
    this.shown = this.ops.length;
  };

  /* Send the outcomes: one step passes. Afterwards this.info says what was read at the start of the next step. */
  Player.prototype.send = function () {
    var link = this.link, rec = this.rec, k, self = this;
    this.shown = this.ops.length;
    this.after = snap(link);
    var used = {};
    rec.ops.forEach(function (o) { used[o.use] = true; });
    for (k = 0; k < this.N; k++) {
      rec.end[k] = used[k] ? 'meter' : rec.discards.indexOf(k) >= 0 ? 'discard' : link.slots[k].occ ? 'carry' : null;
    }
    rec.sent = true;
    if (this.named.length) link.deliverIfVerified(this.named);
    var info = link.end();
    info.results.forEach(function (r) { rec.outcome[r.k] = r.ok ? 'ok' : 'fail'; });
    info.delivered.forEach(function (d) { rec.outcome[d.k] = d.F >= self.thr ? 'deliv' : 'low'; });
    this.steps++;
    this.hist.push(rec);
    this.info = info;
    this.rec = this.openRecord(info.arrivals);
    this.before = snap(link);
    this.ops = [];
    this.named = [];
    this.undo = [];
    this.shown = 0;
    this.sel = null;
    this.hover = null;
  };

  Player.prototype.blocks = function () {           /* the blocks built in this step */
    var out = [], link = this.link;
    link.slots.forEach(function (s, k) {
      if (s.occ && s.pc.length) out.push({ k: k, ops: s.pc.length, p: link.pBlock(s), Fif: 0.25 + link.lam * (s.A - 0.25) });
    });
    return out;
  };

  /* What each memory shows now: gen (generating), pair, kept and used (in a block of this step), pend (sent,
   * not confirmed), deliv, fail. */
  Player.prototype.display = function () {
    var out = [], k, N = this.N, self = this;
    if (this.phase === 'travel') {
      for (k = 0; k < N; k++) {
        var a = this.after[k];
        out.push(!a.occ ? { kind: 'gen' } : a.pend ? { kind: 'pend', F: a.Fif, p: a.p } : { kind: 'pair', F: a.F });
      }
      return out;
    }
    if (this.phase === 'arrive') {
      var info = this.info, now = this.before;
      for (k = 0; k < N; k++) {
        out.push(!now[k].occ ? { kind: 'gen' } : { kind: 'pair', F: now[k].F, fresh: info.arrivals.indexOf(k) >= 0 });
      }
      info.results.forEach(function (r) { if (!r.ok) out[r.k] = { kind: 'fail' }; else if (out[r.k].kind === 'pair') out[r.k].ok = true; });
      info.delivered.forEach(function (d) { out[d.k] = { kind: 'deliv', F: d.F, low: d.F < self.thr }; });
      return out;
    }
    for (k = 0; k < N; k++) out.push(this.before[k].occ ? { kind: 'pair', F: this.before[k].F } : { kind: 'gen' });
    for (var i = 0; i < this.shown; i++) {
      var o = this.ops[i];
      out[o.use] = { kind: 'used', F: this.before[o.use].F };
      out[o.keep] = { kind: 'kept', F: this.before[o.keep].F, Fif: o.Fif, p: o.pBlock };
    }
    this.rec.discards.forEach(function (d) { out[d] = { kind: 'gen' }; });
    return out;
  };

  Player.prototype.treesNow = function () {
    var src = this.phase === 'travel' ? this.after : snap(this.link), out = [];
    src.forEach(function (s, k) {
      if (s.occ && s.tag !== 'o') out.push({ k: k, tag: s.tag, pend: s.pend, F: s.pend ? s.Fif : s.F, p: s.p });
    });
    return out;
  };

  /* ---------- the element ---------- */
  var NODE = { ax: 92, bx: 408, y: 106, R: 86, ring: 52, m: 11.5, g: 15.5 };
  var ANG = [-112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5, 202.5];
  var CIRC = { sw: 150, ws: 12, pad: 8, top: 6, gap: 26 };
  CIRC.lane = 2 * CIRC.pad + 7 * CIRC.ws;
  CIRC.h = CIRC.top + 2 * CIRC.lane + CIRC.gap + 18;

  function polar(cx, cy, r, deg) { var a = deg * Math.PI / 180; return [cx + r * Math.cos(a), cy + r * Math.sin(a)]; }
  function gaugeFrac(F) { return Math.max(0, Math.min(1, (F - 0.7) / 0.3)); }
  function arcPath(cx, cy, r, frac) {
    if (frac >= 0.999) return 'M' + r2(cx) + ' ' + r2(cy - r) + 'A' + r + ' ' + r + ' 0 1 1 ' + r2(cx - 0.01) + ' ' + r2(cy - r);
    var a = (-90 + 360 * frac) * Math.PI / 180;
    return 'M' + r2(cx) + ' ' + r2(cy - r) + 'A' + r + ' ' + r + ' 0 ' + (frac > 0.5 ? 1 : 0) + ' 1 ' + r2(cx + r * Math.cos(a)) + ' ' + r2(cy + r * Math.sin(a));
  }

  var CSS = ":host { display: block; } .dd { --bg: #fafaf8; --surface: #ffffff; --ink: #161616; --text: #3a3a37; --muted: #75746e; --line: #e8e7e2; --line-strong: #cfcdc5; --track: #eceae3; --lane: #edf1f6; --lane-now: #e2e9f2; --accent: #1b7f67; --accent-fill: #e9f3ef; --on-accent: #ffffff; --pair: #1f6fb5; --wait: #c4472b; --wait-fill: #fbf1ee; --pend: #9a6408; --pend-fill: #f9f5ec; --mono: ui-monospace, \"SF Mono\", \"IBM Plex Mono\", Menlo, Consolas, monospace; container-type: inline-size; color: var(--ink); font-family: inherit; font-size: 14px; line-height: 1.5; text-align: left; -webkit-font-smoothing: antialiased; } .dd[data-theme=\"dark\"] { --bg: #121311; --surface: #1a1b19; --ink: #ecebe6; --text: #c9c8c2; --muted: #9a9a93; --line: #2a2b28; --line-strong: #474842; --track: #2b2c29; --lane: #24272b; --lane-now: #2b3037; --accent: #4fc3a1; --accent-fill: #1c2b26; --on-accent: #0e1512; --pair: #74b4ec; --wait: #ee8169; --wait-fill: #2b1f1c; --pend: #e0b055; --pend-fill: #27231a; } @media (prefers-color-scheme: dark) { .dd[data-theme=\"auto\"] { --bg: #121311; --surface: #1a1b19; --ink: #ecebe6; --text: #c9c8c2; --muted: #9a9a93; --line: #2a2b28; --line-strong: #474842; --track: #2b2c29; --lane: #24272b; --lane-now: #2b3037; --accent: #4fc3a1; --accent-fill: #1c2b26; --on-accent: #0e1512; --pair: #74b4ec; --wait: #ee8169; --wait-fill: #2b1f1c; --pend: #e0b055; --pend-fill: #27231a; } } *, *::before, *::after { box-sizing: border-box; } .defs { position: absolute; width: 0; height: 0; overflow: hidden; } button { font: inherit; color: inherit; } button:focus-visible, .scroll:focus-visible, summary:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; } /* choices */ .top { display: flex; flex-wrap: wrap; gap: 14px 28px; align-items: flex-end; } .pick { display: flex; flex-direction: column; align-items: flex-start; gap: 6px; min-width: 0; } .lbl { font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); } .ttl { font-size: 12.5px; line-height: 1.35; color: var(--muted); } .ttl b { margin-right: 7px; font-size: 13px; font-weight: 600; color: var(--ink); } .seg { display: inline-flex; flex-wrap: wrap; gap: 2px; padding: 3px; border: 1px solid var(--line); border-radius: 999px; background: var(--surface); } .seg button { border: 0; background: none; padding: 6px 13px; border-radius: 999px; color: var(--muted); cursor: pointer; white-space: nowrap; transition: color 0.15s ease, background-color 0.15s ease; } .seg button:hover { color: var(--ink); } .seg button[aria-pressed=\"true\"] { background: var(--ink); color: var(--bg); } .presets { display: flex; flex-wrap: wrap; gap: 6px; } .presets button { display: flex; flex-direction: column; align-items: flex-start; gap: 1px; padding: 7px 14px 8px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); cursor: pointer; text-align: left; transition: border-color 0.15s ease, background-color 0.15s ease, color 0.15s ease; } .presets button b { font-size: 14px; font-weight: 600; font-variant-numeric: tabular-nums; white-space: nowrap; } .presets button span { font-size: 11.5px; color: var(--muted); white-space: nowrap; } .presets button:hover { border-color: var(--line-strong); } .presets button[aria-pressed=\"true\"] { background: var(--ink); border-color: var(--ink); color: var(--bg); } .presets button[aria-pressed=\"true\"] span { color: var(--bg); opacity: 0.72; } .ctl { margin-left: auto; } .run { display: flex; align-items: center; gap: 8px; } .spdw { display: inline-flex; align-items: center; gap: 7px; margin-left: 4px; } .spd { font-size: 12.5px; color: var(--muted); } .play, .reset { border: 1px solid var(--line-strong); background: var(--surface); padding: 7px 15px; border-radius: 999px; cursor: pointer; font-weight: 600; } .play { min-width: 74px; } .play:hover, .reset:hover { border-color: var(--accent); color: var(--accent); } .ctl.manual .play, .ctl.manual .spdw { display: none; } .facts { margin: 14px 0 0; font-size: 13.5px; color: var(--text); } .facts b, .foot b { color: var(--ink); font-weight: 600; } .facts .more { display: block; margin-top: 4px; color: var(--muted); } /* where each policy stands in the setting */ .longrun { display: grid; grid-template-columns: minmax(0, 330px) minmax(0, 1fr); gap: 9px 48px; align-items: start; align-content: start; margin: 16px 0 0; padding-top: 14px; border-top: 1px solid var(--line); min-height: 159px; font-size: 13px; line-height: 1.45; color: var(--text); } .longrun .lbl { grid-column: 1 / -1; } .lr-rows { display: grid; grid-template-columns: max-content minmax(60px, 1fr) max-content; gap: 7px 12px; align-items: center; padding-top: 3px; } .lr { display: contents; } .lr-who { white-space: nowrap; } .lr-who b { font-weight: 600; color: var(--ink); } .lr-who i { margin-left: 7px; font-style: normal; font-size: 12px; color: var(--muted); } .lr-b { height: 6px; background: var(--track); border-radius: 3px; overflow: hidden; } .lr-b i { display: block; height: 100%; background: var(--muted); border-radius: 3px; } .lr.me .lr-b i { background: var(--accent); } .lr-n { text-align: right; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; } .lr-note { grid-column: 1 / -1; margin: 2px 0 0; font-size: 12.5px; color: var(--text); } .lr-why { min-width: 0; max-width: 76ch; margin: 0; padding-top: 1px; font-size: 13.5px; line-height: 1.55; color: var(--text); } /* how it works */ .how { margin: 0 0 16px; border-bottom: 1px solid var(--line); } .how summary { cursor: pointer; padding: 0 0 11px; font-weight: 600; font-size: 13.5px; list-style-position: inside; } .how[open] summary { padding-bottom: 14px; } .primer { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px 22px; margin: 2px 0 0; padding: 0; list-style: none; counter-reset: pg; } .primer li { counter-increment: pg; display: flex; flex-direction: column; gap: 8px; min-width: 0; } .primer svg { display: block; width: 100%; max-width: 330px; height: auto; } .primer p { margin: 0; font-size: 13.5px; line-height: 1.5; color: var(--text); } .primer p::before { content: counter(pg) \" \"; font-family: var(--mono); font-size: 11px; color: var(--muted); } .primer p b { font-weight: 600; color: var(--ink); } .lede { margin: 2px 0 18px; font-size: 15px; line-height: 1.55; color: var(--ink); max-width: 86ch; } .how-h { margin: 24px 0 10px; font-size: 13.5px; font-weight: 600; letter-spacing: -0.01em; color: var(--ink); } .lede + .how-h { margin-top: 0; } .how-p { margin: 0 0 14px; font-size: 13.5px; line-height: 1.55; color: var(--text); max-width: 92ch; } .primer.p2 { counter-reset: pg 3; } .how-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px 36px; margin: 20px 0 18px; } .how-grid h4 { margin: 0 0 6px; font-size: 13.5px; font-weight: 600; letter-spacing: -0.01em; color: var(--ink); } .how-grid p { margin: 0; font-size: 13.5px; line-height: 1.55; color: var(--text); max-width: 64ch; } .how-grid p + p { margin-top: 8px; } .how-grid p b { font-weight: 600; color: var(--ink); } /* the two halves */ .lead { margin: 16px 0 10px; padding-top: 14px; border-top: 1px solid var(--line); min-height: 37px; font-size: 14px; color: var(--text); font-variant-numeric: tabular-nums; } .lead { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; } .lead b { min-width: 70px; font-family: var(--mono); font-size: 12.5px; font-weight: 500; color: var(--muted); } .lead .ph { padding: 2px 10px; border: 1px solid var(--line); border-radius: 999px; font-size: 11.5px; color: var(--muted); white-space: nowrap; } .lead .ph.on { background: var(--ink); border-color: var(--ink); color: var(--bg); } .lead .gap { margin-left: 10px; } .halves { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; align-items: start; } .half { min-width: 0; display: flex; flex-direction: column; gap: 8px; padding: 16px 16px 18px; background: var(--surface); border: 1px solid var(--line); border-radius: 16px; } .head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap; } .who { display: flex; flex-direction: column; gap: 1px; } .who .name { font-size: 17px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.25; } .who .sub { font-size: 12px; color: var(--muted); } .score { display: flex; align-items: baseline; gap: 6px; font-size: 12.5px; color: var(--muted); font-variant-numeric: tabular-nums; } .score .n { font-size: 24px; line-height: 1; font-weight: 700; color: var(--ink); letter-spacing: -0.02em; } .score .rate { margin-left: 6px; } .meter { position: relative; height: 6px; margin: 2px 0 20px; background: var(--track); border-radius: 3px; } .meter .fill { position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent); border-radius: 3px; transition: width 0.3s ease; } .meter .tick { position: absolute; top: -4px; width: 2px; height: 14px; margin-left: -1px; background: var(--ink); border-radius: 1px; } .meter .cap { position: absolute; top: 11px; transform: translateX(-50%); font-size: 10.5px; color: var(--muted); white-space: nowrap; } .stage svg { display: block; width: 100%; height: auto; } .bar { display: flex; align-items: center; justify-content: space-between; gap: 8px 14px; flex-wrap: wrap; min-height: 46px; } .say { margin: 0; flex: 1 1 240px; min-width: 0; font-size: 13.5px; color: var(--text); } .acts { display: flex; gap: 8px; } .acts button { border: 1px solid var(--line-strong); background: none; padding: 7px 14px; border-radius: 999px; font-weight: 600; cursor: pointer; white-space: nowrap; } .acts button.main { background: var(--accent); border-color: var(--accent); color: var(--on-accent); min-width: 128px; } .acts button:disabled { opacity: 0.45; cursor: default; } .label { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-top: 10px; font-size: 11px; letter-spacing: 0.07em; text-transform: uppercase; color: var(--muted); } .label span { letter-spacing: 0; text-transform: none; font-size: 11.5px; } .circ { display: grid; grid-template-columns: 30px minmax(0, 1fr); } .rows { display: block; width: 30px; } .scroll { overflow-x: auto; overflow-y: hidden; scrollbar-width: thin; border-radius: 6px; } .strip { display: block; } .trees { display: flex; flex-wrap: wrap; gap: 8px; min-height: 52px; align-content: flex-start; } .tcard { display: flex; align-items: center; gap: 6px; padding: 4px 11px 4px 5px; border: 1px solid var(--line); border-radius: 10px; } .tcard.pend { border-style: dashed; border-color: var(--pend); } .tcard .g svg { display: block; width: 58px; height: 40px; } .tcard .c { display: flex; flex-direction: column; font-size: 11.5px; line-height: 1.35; font-variant-numeric: tabular-nums; } .tcard .c b { font-weight: 600; } .tcard .c span { color: var(--muted); } .tcard.pend .c span { color: var(--pend); } .tally { display: flex; flex-direction: column; gap: 3px; min-height: 132px; } .trow { display: grid; grid-template-columns: 58px minmax(0, max-content) minmax(36px, 1fr) 38px; gap: 10px; align-items: center; } .trow .g svg { display: block; width: 58px; height: 34px; } .trow code { font-family: var(--mono); font-size: 11.5px; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; } .trow code i { font-style: normal; color: var(--wait); } .trow .b { height: 6px; background: var(--track); border-radius: 3px; overflow: hidden; } .trow .b i { display: block; height: 100%; background: var(--pair); border-radius: 3px; transition: width 0.3s ease; } .trow .s { text-align: right; font-size: 12.5px; font-weight: 600; font-variant-numeric: tabular-nums; } .none { margin: 0; font-size: 12.5px; color: var(--muted); } .foot { margin: 16px 0 0; font-size: 12.5px; color: var(--muted); max-width: 104ch; } .foot span { display: block; } .foot span + span { margin-top: 5px; } .foot a { color: var(--accent); text-decoration: none; border-bottom: 1px solid transparent; white-space: nowrap; } .foot a:hover { border-bottom-color: var(--accent); } /* drawing: nodes and memories */ svg text { font-family: inherit; fill: var(--ink); } .node { fill: var(--lane); stroke: var(--line-strong); stroke-width: 1.2; } .node-t { font-size: 12px; font-weight: 600; } .link-t { font-size: 10px; fill: var(--muted); } .fiber { stroke: var(--ink); stroke-width: 2.2; stroke-linecap: round; } .mem .hit { fill: transparent; } .mem .track { fill: none; stroke: var(--track); stroke-width: 3; } .mem .core { fill: var(--surface); stroke: var(--pair); stroke-width: 1.6; } .mem .idx { font-family: var(--mono); font-size: 9.5px; } .mem .arc { fill: none; stroke: var(--pair); stroke-width: 3; stroke-linecap: round; } .mem .goal { stroke: var(--ink); stroke-width: 1.5; stroke-linecap: round; } .m-gen .core { stroke: var(--line-strong); stroke-width: 1.2; stroke-dasharray: 2 2.8; } .m-gen .idx { fill: var(--muted); } .m-gen .track { opacity: 0.55; } .m-used .core, .m-used .idx { opacity: 0.5; } .m-used .arc { opacity: 0.4; } .mem .arc.ghost { stroke: var(--pend); stroke-dasharray: 2.6 2.8; stroke-linecap: butt; } .m-pend .core { fill: var(--pend-fill); stroke: var(--pend); stroke-dasharray: 3.6 2.6; } .m-deliv .core { fill: var(--accent-fill); stroke: var(--accent); stroke-width: 2.2; } .m-deliv .arc { stroke: var(--accent); } .m-deliv.low .core { fill: var(--surface); stroke: var(--muted); } .m-deliv.low .arc { stroke: var(--muted); } .m-fail .core { fill: var(--wait-fill); stroke: var(--wait); } .cross { stroke: var(--wait); stroke-width: 1.9; stroke-linecap: round; fill: none; } .tickmark { stroke: var(--accent); stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; fill: none; } .mirror { opacity: 0.42; filter: saturate(0.55); } .mem.can { cursor: pointer; } .mem.can:hover .core { stroke-width: 2.6; } .mem.sel .core { stroke: var(--accent); stroke-width: 2.8; } .mem:focus { outline: none; } .mem:focus-visible .hit { stroke: var(--accent); stroke-width: 1.6; } .mem .arc.hint { stroke: var(--accent); opacity: 0.6; } .hint-t { font-size: 9.5px; font-weight: 700; fill: var(--accent); font-variant-numeric: tabular-nums; } .chance { font-size: 9.5px; font-weight: 700; fill: var(--pend); font-variant-numeric: tabular-nums; } .bond { fill: none; stroke: var(--accent); stroke-width: 2; stroke-linecap: round; } .bond-d { fill: var(--accent); } .msg { fill: none; stroke: var(--wait); stroke-width: 1.3; stroke-dasharray: 4.5 3.5; } .travel .msg { stroke-width: 1.7; animation: flow 0.7s linear infinite; } .msg-t { font-size: 10px; font-weight: 600; fill: var(--wait); } .ahr { fill: var(--wait); } .ahk { fill: var(--ink); } .mem.fresh .core, .mem.fresh .arc { animation: pop 0.4s ease-out both; transform-box: fill-box; transform-origin: center; } .m-deliv .core { animation: glow 0.6s ease-out both; transform-box: fill-box; transform-origin: center; } @keyframes flow { to { stroke-dashoffset: -16; } } @keyframes pop { from { opacity: 0; transform: scale(0.5); } } @keyframes glow { from { transform: scale(1.35); } } /* drawing: the circuit */ .lane { fill: var(--lane); } .now .lane { fill: var(--lane-now); } .guide { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 1.2 3.2; stroke-linecap: round; } .step-t { font-size: 9.5px; fill: var(--muted); } .lane-t { font-size: 9.5px; fill: var(--muted); } .row-t { font-family: var(--mono); font-size: 8.5px; fill: var(--muted); } .wire { stroke: var(--ink); stroke-width: 1; } .bub { fill: var(--surface); stroke: var(--pair); stroke-width: 1.4; } .gate { fill: none; stroke: var(--ink); stroke-width: 1.1; stroke-linecap: round; } .gate-d { fill: var(--ink); } .gate-t { fill: var(--surface); stroke: var(--ink); stroke-width: 1.1; } .mtr { fill: var(--surface); stroke: var(--ink); stroke-width: 0.9; } .mtr-a { fill: none; stroke: var(--ink); stroke-width: 0.8; stroke-linecap: round; } .done { fill: var(--accent); } .done.low { fill: var(--muted); } .gone { stroke: var(--muted); stroke-width: 1.2; stroke-linecap: round; } /* drawing: trees */ .leaf { fill: var(--surface); stroke: var(--pair); stroke-width: 1.3; } .edge { stroke: var(--ink); stroke-width: 1.1; stroke-linecap: round; } .edge-p { stroke-dasharray: 2 2.2; } .op-n { fill: var(--pair); } .op-p { fill: var(--surface); stroke: var(--pair); stroke-width: 1.3; stroke-dasharray: 1.8 1.6; } .op-w { fill: var(--wait); stroke: var(--surface); stroke-width: 1; } /* drawing: the three pictures */ .p-node { fill: var(--lane); } .p-lab { font-size: 11px; fill: var(--muted); } .p-val { font-size: 12px; font-weight: 600; } .p-pair { fill: none; stroke: var(--pair); stroke-width: 2; stroke-linecap: round; } .p-pair.strong { stroke-width: 3.2; } .p-pair.faint { opacity: 0.5; } .p-mem { fill: var(--surface); stroke: var(--pair); stroke-width: 1.5; } .p-mem.off { stroke: var(--muted); stroke-width: 1.1; stroke-dasharray: 2 2.4; } .p-num { font-family: var(--mono); font-size: 8.5px; } .p-arrow { fill: none; stroke: var(--ink); stroke-width: 1.4; } .p-msg { fill: none; stroke: var(--wait); stroke-width: 1.5; stroke-dasharray: 5 3.5; } .p-red { font-size: 10.5px; font-weight: 600; fill: var(--wait); } .p-x { stroke: var(--wait); stroke-width: 1.8; stroke-linecap: round; } .p-sm { font-size: 9.5px; fill: var(--muted); font-variant-numeric: tabular-nums; } .p-span { fill: none; stroke: var(--muted); stroke-width: 1; } .p-edge { fill: none; stroke: var(--ink); stroke-width: 1.3; stroke-linecap: round; } .p-block { fill: var(--pend-fill); stroke: var(--pend); stroke-width: 1.2; stroke-dasharray: 3.6 2.6; } .p-blk { font-size: 10.5px; font-weight: 600; fill: var(--pend); } @container (max-width: 900px) { .halves { grid-template-columns: minmax(0, 1fr); } .primer, .how-grid { grid-template-columns: minmax(0, 1fr); } .longrun { grid-template-columns: minmax(0, 1fr); gap: 12px; min-height: 0; } .lr-rows { max-width: 420px; } .ctl { margin-left: 0; } .run { flex-wrap: wrap; } .run .seg { flex-wrap: nowrap; } } @media (prefers-reduced-motion: reduce) { .travel .msg, .mem.fresh .core, .mem.fresh .arc, .m-deliv .core { animation: none; } .meter .fill, .trow .b i { transition: none; } }";
  var PRIMER = "<p class=\"lede\">This demo runs the model of our paper. Two nodes generate entangled pairs over a link and distill them until they reach the fidelity that an application requires. A scheduling policy chooses the distillation operations. The page shows the same link twice: on the left it is scheduled by one of two baseline policies, or by hand if you want to try, and on the right by the policy we trained with reinforcement learning.</p><h4 class=\"how-h\">The model</h4><ol class=\"primer\"><li><svg viewBox=\"0 0 320 132\" role=\"img\" aria-label=\"Two nodes with eight numbered memories each. Memories 2, 3, 5 and 8 hold an entangled pair; the other memories are empty and attempt to generate one\"><rect class=\"p-node\" x=\"8\" y=\"2\" width=\"64\" height=\"128\" rx=\"9\"/><rect class=\"p-node\" x=\"248\" y=\"2\" width=\"64\" height=\"128\" rx=\"9\"/><text class=\"p-lab\" transform=\"translate(23,66) rotate(-90)\" text-anchor=\"middle\">node A</text><text class=\"p-lab\" transform=\"translate(304,66) rotate(-90)\" text-anchor=\"middle\">node B</text><path class=\"p-pair\" d=\"M55 27.8H265M55 43.1H265M55 73.7H265M55 119.6H265\"/><circle class=\"p-mem off\" cx=\"55\" cy=\"12.5\" r=\"5.2\"/><circle class=\"p-mem off\" cx=\"265\" cy=\"12.5\" r=\"5.2\"/><circle class=\"p-mem\" cx=\"55\" cy=\"27.8\" r=\"5.2\"/><circle class=\"p-mem\" cx=\"265\" cy=\"27.8\" r=\"5.2\"/><circle class=\"p-mem\" cx=\"55\" cy=\"43.1\" r=\"5.2\"/><circle class=\"p-mem\" cx=\"265\" cy=\"43.1\" r=\"5.2\"/><circle class=\"p-mem off\" cx=\"55\" cy=\"58.4\" r=\"5.2\"/><circle class=\"p-mem off\" cx=\"265\" cy=\"58.4\" r=\"5.2\"/><circle class=\"p-mem\" cx=\"55\" cy=\"73.7\" r=\"5.2\"/><circle class=\"p-mem\" cx=\"265\" cy=\"73.7\" r=\"5.2\"/><circle class=\"p-mem off\" cx=\"55\" cy=\"89\" r=\"5.2\"/><circle class=\"p-mem off\" cx=\"265\" cy=\"89\" r=\"5.2\"/><circle class=\"p-mem off\" cx=\"55\" cy=\"104.3\" r=\"5.2\"/><circle class=\"p-mem off\" cx=\"265\" cy=\"104.3\" r=\"5.2\"/><circle class=\"p-mem\" cx=\"55\" cy=\"119.6\" r=\"5.2\"/><circle class=\"p-mem\" cx=\"265\" cy=\"119.6\" r=\"5.2\"/><text class=\"row-t\" x=\"38\" y=\"15.5\" text-anchor=\"middle\">1</text><text class=\"row-t\" x=\"282\" y=\"15.5\" text-anchor=\"middle\">1</text><text class=\"row-t\" x=\"38\" y=\"30.8\" text-anchor=\"middle\">2</text><text class=\"row-t\" x=\"282\" y=\"30.8\" text-anchor=\"middle\">2</text><text class=\"row-t\" x=\"38\" y=\"46.1\" text-anchor=\"middle\">3</text><text class=\"row-t\" x=\"282\" y=\"46.1\" text-anchor=\"middle\">3</text><text class=\"row-t\" x=\"38\" y=\"61.4\" text-anchor=\"middle\">4</text><text class=\"row-t\" x=\"282\" y=\"61.4\" text-anchor=\"middle\">4</text><text class=\"row-t\" x=\"38\" y=\"76.7\" text-anchor=\"middle\">5</text><text class=\"row-t\" x=\"282\" y=\"76.7\" text-anchor=\"middle\">5</text><text class=\"row-t\" x=\"38\" y=\"92\" text-anchor=\"middle\">6</text><text class=\"row-t\" x=\"282\" y=\"92\" text-anchor=\"middle\">6</text><text class=\"row-t\" x=\"38\" y=\"107.3\" text-anchor=\"middle\">7</text><text class=\"row-t\" x=\"282\" y=\"107.3\" text-anchor=\"middle\">7</text><text class=\"row-t\" x=\"38\" y=\"122.6\" text-anchor=\"middle\">8</text><text class=\"row-t\" x=\"282\" y=\"122.6\" text-anchor=\"middle\">8</text><text class=\"p-lab\" x=\"160\" y=\"22\" text-anchor=\"middle\">entangled pairs</text><text class=\"p-sm\" x=\"160\" y=\"100\" text-anchor=\"middle\">empty: attempting generation</text></svg><p><b>The link.</b> Two nodes with eight quantum memories each. An entangled pair occupies the memories with the same number at the two nodes. Every empty memory attempts to generate a pair, and a new pair has a lower fidelity than the application requires.</p></li><li><svg viewBox=\"0 0 320 132\" role=\"img\" aria-label=\"Two pairs of fidelity 0.90 become one pair of fidelity 0.92 with probability 87 percent\"><path class=\"p-pair\" d=\"M24 46H104\"/><path class=\"p-pair\" d=\"M24 82H104\"/><circle class=\"p-mem\" cx=\"24\" cy=\"46\" r=\"5.5\"/><circle class=\"p-mem\" cx=\"104\" cy=\"46\" r=\"5.5\"/><circle class=\"p-mem\" cx=\"24\" cy=\"82\" r=\"5.5\"/><circle class=\"p-mem\" cx=\"104\" cy=\"82\" r=\"5.5\"/><text class=\"p-val\" x=\"64\" y=\"37\" text-anchor=\"middle\">0.90</text><text class=\"p-val\" x=\"64\" y=\"73\" text-anchor=\"middle\">0.90</text><path class=\"p-arrow\" d=\"M124 64H194\" marker-end=\"url(#ahk)\"/><text class=\"p-lab\" x=\"159\" y=\"55\" text-anchor=\"middle\">succeeds 87%</text><path class=\"p-pair strong\" d=\"M214 64H296\"/><circle class=\"p-mem\" cx=\"214\" cy=\"64\" r=\"5.5\"/><circle class=\"p-mem\" cx=\"296\" cy=\"64\" r=\"5.5\"/><text class=\"p-val\" x=\"255\" y=\"54\" text-anchor=\"middle\">0.92</text><path class=\"p-x\" d=\"M101 109l8 8M109 109l-8 8\"/><text class=\"p-red\" x=\"116\" y=\"117\">fails 13%: both pairs are lost</text></svg><p><b>A distillation operation.</b> We use the two-to-one DEJMPS protocol. An operation consumes two stored pairs and, if it succeeds, leaves one pair of higher fidelity. If it fails, both pairs are lost.</p></li><li><svg viewBox=\"0 0 320 132\" role=\"img\" aria-label=\"The circuit of one operation: at each node a gate on memories 1 and 2 and a measurement of memory 2; the outcomes cross the link in one step, and only then do the nodes know whether the operation succeeded\"><rect class=\"lane\" x=\"40\" y=\"12\" width=\"270\" height=\"40\" rx=\"3\"/><rect class=\"lane\" x=\"40\" y=\"74\" width=\"270\" height=\"40\" rx=\"3\"/><text class=\"lane-t\" transform=\"translate(11,32) rotate(-90)\" text-anchor=\"middle\">node A</text><text class=\"lane-t\" transform=\"translate(11,94) rotate(-90)\" text-anchor=\"middle\">node B</text><text class=\"row-t\" x=\"31\" y=\"30\" text-anchor=\"middle\">1</text><text class=\"row-t\" x=\"31\" y=\"46\" text-anchor=\"middle\">2</text><text class=\"row-t\" x=\"31\" y=\"86\" text-anchor=\"middle\">2</text><text class=\"row-t\" x=\"31\" y=\"102\" text-anchor=\"middle\">1</text><line class=\"wire\" x1=\"56\" y1=\"27\" x2=\"310\" y2=\"27\"/><line class=\"wire\" x1=\"56\" y1=\"43\" x2=\"120\" y2=\"43\"/><line class=\"wire\" x1=\"56\" y1=\"83\" x2=\"120\" y2=\"83\"/><line class=\"wire\" x1=\"56\" y1=\"99\" x2=\"310\" y2=\"99\"/><circle class=\"bub\" cx=\"56\" cy=\"27\" r=\"3.8\"/><circle class=\"bub\" cx=\"56\" cy=\"43\" r=\"3.8\"/><circle class=\"bub\" cx=\"56\" cy=\"83\" r=\"3.8\"/><circle class=\"bub\" cx=\"56\" cy=\"99\" r=\"3.8\"/><line class=\"gate\" x1=\"90\" y1=\"27\" x2=\"90\" y2=\"43\"/><circle class=\"gate-d\" cx=\"90\" cy=\"27\" r=\"2.5\"/><circle class=\"gate-t\" cx=\"90\" cy=\"43\" r=\"4.4\"/><path class=\"gate\" d=\"M85.6 43H94.4M90 38.6V47.4\"/><line class=\"gate\" x1=\"90\" y1=\"99\" x2=\"90\" y2=\"83\"/><circle class=\"gate-d\" cx=\"90\" cy=\"99\" r=\"2.5\"/><circle class=\"gate-t\" cx=\"90\" cy=\"83\" r=\"4.4\"/><path class=\"gate\" d=\"M85.6 83H94.4M90 78.6V87.4\"/><rect class=\"mtr\" x=\"120\" y=\"37.5\" width=\"14\" height=\"11\" rx=\"1.6\"/><path class=\"mtr-a\" d=\"M123 46A4 4 0 0 1 131 46M127 46.2L130.2 40.2\"/><rect class=\"mtr\" x=\"120\" y=\"77.5\" width=\"14\" height=\"11\" rx=\"1.6\"/><path class=\"mtr-a\" d=\"M123 86A4 4 0 0 1 131 86M127 86.2L130.2 80.2\"/><text class=\"p-red\" x=\"88\" y=\"66.5\" text-anchor=\"middle\">outcomes</text><path class=\"p-msg\" d=\"M140 53L231 73\" marker-end=\"url(#ahr)\"/><path class=\"p-msg\" d=\"M140 73L231 53\" marker-end=\"url(#ahr)\"/><line class=\"guide\" x1=\"238\" y1=\"6\" x2=\"238\" y2=\"120\"/><text class=\"p-sm\" x=\"187\" y=\"22\" text-anchor=\"middle\">success unknown</text><text class=\"p-sm\" x=\"274\" y=\"22\" text-anchor=\"middle\">known</text><path class=\"p-span\" d=\"M127 123.5H238M127 120.5V126.5M238 120.5V126.5\"/><text class=\"p-lab\" x=\"268\" y=\"127\" text-anchor=\"middle\">one step</text></svg><p><b>The measurement outcomes.</b> At each node, the operation applies a two-qubit gate to the two memories and measures one of them. It has succeeded if the two measurement outcomes agree, so the nodes send them to each other. A message takes one step to cross the link; the step is the time unit of the model.</p></li></ol><h4 class=\"how-h\">The scheduling problem</h4><p class=\"how-p\">A pair reaches the required fidelity through several operations. While the outcomes of an operation cross the link, the stored pairs decohere, idle in their memories. This raises a dilemma: wait for the outcomes before using the output, or use it at once, at the risk of needlessly consuming further pairs if the operation has failed. A scheduling policy decides in every step which stored pairs to distill together and, for each operation, whether to wait for its outcomes. The goal is the highest throughput: the pairs delivered per step with at least the required fidelity.</p><h4 class=\"how-h\">The policies</h4><p class=\"how-p\">The operations behind one delivered pair form its distillation tree. The pictures show a tree on four new pairs, with three operations, scheduled in three ways. Time runs from left to right, and the outcomes of a step arrive at the dotted line.</p><ol class=\"primer p2\"><li><svg viewBox=\"0 0 320 132\" role=\"img\" aria-label=\"Nested: two operations in step 1, each on two new pairs; their outcomes arrive; the third operation in step 2 on the two outputs; the pair is delivered when its outcomes arrive\"><line class=\"guide\" x1=\"146\" y1=\"16\" x2=\"146\" y2=\"110\"/><line class=\"guide\" x1=\"254\" y1=\"16\" x2=\"254\" y2=\"110\"/><text class=\"p-sm\" x=\"146\" y=\"10\" text-anchor=\"middle\">outcomes arrive</text><path class=\"p-edge\" d=\"M44 26L96 36M44 46L96 36M44 76L96 86M44 96L96 86M96 36L200 61M96 86L200 61M200 61H272\"/><circle class=\"leaf\" cx=\"44\" cy=\"26\" r=\"5\"/><circle class=\"leaf\" cx=\"44\" cy=\"46\" r=\"5\"/><circle class=\"leaf\" cx=\"44\" cy=\"76\" r=\"5\"/><circle class=\"leaf\" cx=\"44\" cy=\"96\" r=\"5\"/><path class=\"op-w\" d=\"M89 36L96 29L103 36L96 43Z\"/><path class=\"op-w\" d=\"M89 86L96 79L103 86L96 93Z\"/><path class=\"op-w\" d=\"M193 61L200 54L207 61L200 68Z\"/><circle class=\"done\" cx=\"272\" cy=\"61\" r=\"4.5\"/><text class=\"p-sm\" x=\"284\" y=\"79\" text-anchor=\"middle\">delivered</text><text class=\"p-lab\" x=\"90\" y=\"127\" text-anchor=\"middle\">step 1</text><text class=\"p-lab\" x=\"200\" y=\"127\" text-anchor=\"middle\">step 2</text></svg><p><b>Nested</b> always waits. It applies two operations, waits one step for their outcomes, and applies the third to their outputs. If one of the two has failed, the output of the other is kept.</p></li><li><svg viewBox=\"0 0 320 132\" role=\"img\" aria-label=\"Optimistic: the three operations in step 1 as one block; the pair is delivered when the outcomes arrive\"><line class=\"guide\" x1=\"146\" y1=\"16\" x2=\"146\" y2=\"110\"/><line class=\"guide\" x1=\"254\" y1=\"16\" x2=\"254\" y2=\"110\"/><rect class=\"p-block\" x=\"70\" y=\"20\" width=\"66\" height=\"82\" rx=\"10\"/><text class=\"p-blk\" x=\"103\" y=\"14\" text-anchor=\"middle\">one block</text><path class=\"p-edge\" d=\"M44 26L86 36M44 46L86 36M44 76L86 86M44 96L86 86M86 36L120 61M86 86L120 61M120 61H164\"/><circle class=\"leaf\" cx=\"44\" cy=\"26\" r=\"5\"/><circle class=\"leaf\" cx=\"44\" cy=\"46\" r=\"5\"/><circle class=\"leaf\" cx=\"44\" cy=\"76\" r=\"5\"/><circle class=\"leaf\" cx=\"44\" cy=\"96\" r=\"5\"/><circle class=\"op-n\" cx=\"86\" cy=\"36\" r=\"5.4\"/><circle class=\"op-n\" cx=\"86\" cy=\"86\" r=\"5.4\"/><path class=\"op-w\" d=\"M113 61L120 54L127 61L120 68Z\"/><circle class=\"done\" cx=\"164\" cy=\"61\" r=\"4.5\"/><text class=\"p-sm\" x=\"174\" y=\"64.5\">delivered</text><text class=\"p-lab\" x=\"90\" y=\"127\" text-anchor=\"middle\">step 1</text><text class=\"p-lab\" x=\"200\" y=\"127\" text-anchor=\"middle\">step 2</text></svg><p><b>Optimistic</b> never waits. It applies the three operations in one step, so the third is applied even if one of the first two has failed, and then all four pairs are lost. Operations that produce one pair without waiting for each other's outcomes form a <b>block</b>.</p></li><li><svg viewBox=\"0 0 320 132\" role=\"img\" aria-label=\"The learned policy: one operation in step 1 on two new pairs; its outcomes arrive; in step 2 two new pairs arrive and the other two operations are applied as one block; the pair is delivered when the outcomes arrive\"><line class=\"guide\" x1=\"146\" y1=\"16\" x2=\"146\" y2=\"110\"/><line class=\"guide\" x1=\"254\" y1=\"16\" x2=\"254\" y2=\"110\"/><rect class=\"p-block\" x=\"184\" y=\"45\" width=\"60\" height=\"57\" rx=\"10\"/><text class=\"p-blk\" x=\"214\" y=\"39\" text-anchor=\"middle\">one block</text><path class=\"p-edge\" d=\"M44 26L96 36M44 46L96 36M96 36L228 61M166 76L198 86M166 96L198 86M198 86L228 61M228 61H272\"/><circle class=\"leaf\" cx=\"44\" cy=\"26\" r=\"5\"/><circle class=\"leaf\" cx=\"44\" cy=\"46\" r=\"5\"/><circle class=\"leaf\" cx=\"166\" cy=\"76\" r=\"5\"/><circle class=\"leaf\" cx=\"166\" cy=\"96\" r=\"5\"/><path class=\"op-w\" d=\"M89 36L96 29L103 36L96 43Z\"/><circle class=\"op-n\" cx=\"198\" cy=\"86\" r=\"5.4\"/><path class=\"op-w\" d=\"M221 61L228 54L235 61L228 68Z\"/><circle class=\"done\" cx=\"272\" cy=\"61\" r=\"4.5\"/><text class=\"p-sm\" x=\"284\" y=\"79\" text-anchor=\"middle\">delivered</text><text class=\"p-sm\" x=\"174\" y=\"114\" text-anchor=\"middle\">new pairs</text><text class=\"p-lab\" x=\"90\" y=\"127\" text-anchor=\"middle\">step 1</text><text class=\"p-lab\" x=\"218\" y=\"127\" text-anchor=\"middle\">step 2</text></svg><p><b>The learned policy</b> can wait for some outcomes and not for others, depending on the pairs in memory. Here it waits for the first operation and, when two new pairs arrive, applies the other two operations as one block.</p></li></ol><div class=\"how-grid\"><div><h4>Reading the display</h4><p>The ring around a memory shows the fidelity of its pair, and the mark on the ring shows the required fidelity. A memory drawn dashed holds the output of a block whose outcomes are still crossing the link; the number next to it is the probability that the block succeeds. In a tree, a red diamond is an operation whose outcomes were waited for, and a blue dot is one whose output was used at once. In the string of a tree, o is a new pair, (xy) the output of an operation on x and y, and [x] a pair used only after its outcomes were known.</p></div><div><h4>Play it yourself</h4><p>Select <b>You</b> under &ldquo;Policy on the left side&rdquo; to schedule that side yourself. Click two memories of node A to apply an operation to their pairs; the output stays in the memory you clicked first. To continue without waiting, click that memory again together with another one: the new operation joins the same block. <b>Send outcomes</b> ends the step. Until then you can undo, and you can discard a selected pair. A pair at or above the required fidelity is delivered when its outcomes arrive; a pair below it stays in memory for further operations.</p></div></div>";

  class Demo extends HTMLElement {
    static get observedAttributes() { return ['theme']; }
  }

  Demo.prototype.disconnectedCallback = function () { this.gen++; };

  Demo.prototype.connectedCallback = function () {
    if (this.ready) { this.start(); return; }
    this.ready = true;
    this.gen = 0;
    this.preset = PRESETS[0];
    this.modeId = 'nested';
    this.speed = 1;
    this.paused = false;
    this.seed = (Math.random() * 4294967296) >>> 0;
    this.reduced = !!(root.matchMedia && root.matchMedia('(prefers-reduced-motion: reduce)').matches);
    var sh = this.attachShadow({ mode: 'open' });
    sh.innerHTML = '<style>' + CSS + '</style>' + this.template();
    this.sh = sh;
    this.box = sh.querySelector('.dd');
    this.applyTheme();
    this.bind();
    var self = this;
    this.visible = true;
    if (root.IntersectionObserver) {
      this.visible = false;
      new IntersectionObserver(function (es) {
        /* Several changes can arrive in one call, oldest first (off screen, then on screen, when the page jumps to
         * the demo right after loading). Only the last one says where the demo is now. */
        self.visible = es[es.length - 1].isIntersecting;
        if (self.visible && self.wake) { var w = self.wake; self.wake = null; w(); }
      }, { threshold: 0.05 }).observe(this);
    }
    this.start();
  };

  Demo.prototype.attributeChangedCallback = function () { if (this.box) this.applyTheme(); };
  Demo.prototype.applyTheme = function () {
    var t = this.getAttribute('theme');
    this.box.setAttribute('data-theme', t === 'dark' || t === 'light' ? t : 'auto');
  };

  Demo.prototype.template = function () {
    function seg(cls, items, on) {
      return '<div class="seg ' + cls + '" role="group">' + items.map(function (it) {
        return '<button type="button" data-v="' + it.id + '" aria-pressed="' + (it.id === on) + '">' + esc(it.name) + '</button>';
      }).join('') + '</div>';
    }
    var presets = '<div class="presets" role="group" aria-label="Setting">' + PRESETS.map(function (p) {
      return '<button type="button" data-v="' + p.id + '" aria-pressed="' + (p.id === this.preset.id) + '"><b>' + p.F0.toFixed(2) + ' → ' + p.Fthr.toFixed(2) + '</b><span>' +
        (p.q === 1 ? 'every generation attempt succeeds' : p.q === 0.5 ? 'half of the generation attempts succeed' : '1 generation attempt in ' + Math.round(1 / p.q) + ' succeeds') + '</span></button>';
    }, this).join('') + '</div>';
    function half(side, title) {
      return '<section class="half" data-side="' + side + '">' +
        '<div class="head"><div class="who">' + title + '</div><div class="score"><b class="n">0</b><span class="unit">delivered</span><span class="rate"></span></div></div>' +
        '<div class="meter" aria-hidden="true"><i class="fill"></i><i class="tick"></i><span class="cap"></span></div>' +
        '<div class="stage"><svg class="nodes" viewBox="0 0 500 228" role="group" aria-label="The two nodes and their memories"></svg></div>' +
        '<div class="bar"><p class="say" aria-live="polite"></p><div class="acts"></div></div>' +
        '<div class="label">The circuit, step by step <span>scroll left for earlier steps</span></div>' +
        '<div class="circ"><svg class="rows" viewBox="0 0 30 ' + CIRC.h + '" aria-hidden="true"></svg><div class="scroll" tabindex="0" aria-label="The circuit of the link over time"><svg class="strip" height="' + CIRC.h + '"></svg></div></div>' +
        '<div class="label">Trees in memory now</div><div class="trees"></div>' +
        '<div class="label">Trees delivered <span class="kinds"></span></div><div class="tally"></div>' +
        '</section>';
    }
    return '<div class="dd" data-theme="auto">' +
      '<svg class="defs" aria-hidden="true"><defs>' +
      '<marker id="ahr" viewBox="0 0 10 10" refX="8.2" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="ahr" d="M0 0.6L10 5L0 9.4z"/></marker>' +
      '<marker id="ahk" viewBox="0 0 10 10" refX="8.2" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto-start-reverse"><path class="ahk" d="M0 0.6L10 5L0 9.4z"/></marker>' +
      '</defs></svg>' +
      '<details class="how"><summary>How it works</summary>' + PRIMER + '</details>' +
      '<div class="top">' +
      '<div class="pick"><div class="ttl"><b>Setting</b> (pick one of the three presets: fidelity of a new pair → required fidelity)</div>' + presets + '</div>' +
      '<div class="pick"><div class="ttl"><b>Policy on the left side</b> (the right side runs the learned policy)</div>' + seg('modes', MODES, this.modeId) + '</div>' +
      '<div class="pick ctl"><div class="ttl"><b>Simulation</b></div><div class="run"><button type="button" class="play" aria-label="Pause" title="Pause or resume the simulation">Pause</button>' +
      '<span class="spdw"><span class="spd">Speed</span>' + seg('speeds', SPEEDS, this.speed) + '</span>' +
      '<button type="button" class="reset" title="Start a new run with new random numbers">Reset</button></div></div>' +
      '</div>' +
      '<p class="facts"></p>' +
      '<div class="longrun"></div>' +
      '<div class="lead" aria-live="polite"></div>' +
      '<div class="halves">' + half('L', '<span class="name"></span><span class="sub"></span>') + half('R', '<span class="name">Learned policy</span><span class="sub">trained with reinforcement learning</span>') + '</div>' +
      '<p class="foot"></p>' +
      '</div>';
  };

  Demo.prototype.bind = function () {
    var self = this, sh = this.sh;
    function segClick(cls, fn) {
      sh.querySelector('.' + cls).addEventListener('click', function (e) {
        var b = e.target.closest('button');
        if (!b) return;
        Array.prototype.forEach.call(this.querySelectorAll('button'), function (x) { x.setAttribute('aria-pressed', x === b); });
        fn(b.getAttribute('data-v'));
      });
    }
    segClick('presets', function (v) { self.preset = PRESETS.filter(function (p) { return p.id === v; })[0]; self.start(); });
    segClick('modes', function (v) { self.modeId = v; self.start(); });
    segClick('speeds', function (v) { self.speed = +v; self.kick(); });
    sh.querySelector('.play').addEventListener('click', function () { self.paused = !self.paused; self.renderRun(); self.kick(); });
    sh.querySelector('.reset').addEventListener('click', function () { self.seed = (Math.random() * 4294967296) >>> 0; self.start(); });
    /* "How it works" is closed whenever the page loads; the visitor opens it, and nothing is remembered */
    var left = sh.querySelector('.half[data-side="L"]');
    left.querySelector('.nodes').addEventListener('click', function (e) {
      var g = e.target.closest('[data-k]');
      if (g) self.onMemory(+g.getAttribute('data-k'));
    });
    left.querySelector('.nodes').addEventListener('keydown', function (e) {
      var g = e.target.closest('[data-k]');
      if (g && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); self.onMemory(+g.getAttribute('data-k')); }
    });
    left.querySelector('.nodes').addEventListener('mouseover', function (e) {
      var g = e.target.closest('[data-k]'), L = self.L;
      if (!L || L.mode !== 'you') return;
      var k = g ? +g.getAttribute('data-k') : null;
      if (k !== L.hover) { L.hover = k; self.renderSay(L); }
    });
    left.querySelector('.acts').addEventListener('click', function (e) {
      var b = e.target.closest('button');
      if (!b || !self.L) return;
      var act = b.getAttribute('data-act'), L = self.L;
      if (act === 'undo') { L.userUndo(); L.sel = null; self.renderHalf(L); }
      else if (act === 'discard') { if (L.sel != null) L.userDiscard(L.sel); L.sel = null; self.renderHalf(L); }
      else if (act === 'send' && self.userDone) { var d = self.userDone; self.userDone = null; d(); }
    });
    Array.prototype.forEach.call(sh.querySelectorAll('.scroll'), function (sc) {
      sc.addEventListener('scroll', function () { sc._free = sc.scrollLeft + sc.clientWidth < sc.scrollWidth - 12; });
    });
  };

  /* ---------- running ---------- */
  Demo.prototype.kick = function () { if (this.wakeSleep) { var w = this.wakeSleep; this.wakeSleep = null; w(); } };

  /* Wait for a part of the animation. Shorter at higher speed, nothing at Max, and it holds while paused or
   * while the demo is off screen. */
  Demo.prototype.pause = function (ms, g, always) {
    var self = this;
    return new Promise(function (done) {
      function go() {
        if (g !== self.gen) return done();
        if (!self.visible) { self.wake = go; return; }
        if (self.paused && !always) { self.wakeSleep = go; return; }
        var fast = self.speed === 0 && !always && !self.manual();
        if (fast) {                                   /* a few steps per frame, then let the page draw */
          self.burst = (self.burst || 0) + 1;
          if (self.burst < 6 && root.performance.now() - self.lastFrame < 12) return done();
          self.burst = 0;
          return root.requestAnimationFrame(function () { self.lastFrame = root.performance.now(); done(); });
        }
        var t = self.reduced ? Math.min(ms, 60) : ms / (self.manual() ? 2.2 : (self.speed || 1));
        var timer = root.setTimeout(function () { self.wakeSleep = null; done(); }, t);
        self.wakeSleep = function () { root.clearTimeout(timer); go(); };
      }
      go();
    });
  };
  Demo.prototype.manual = function () { return this.modeId === 'you'; };
  Demo.prototype.quick = function () { return this.speed === 0 && !this.manual(); };

  Demo.prototype.start = function () {
    var self = this, g = ++this.gen, id = this.preset.id;
    this.userDone = null;
    if (this.manual()) this.paused = false;
    this.L = this.R = null;
    this.lastFrame = 0;
    this.renderTop();
    this.setNote('Loading the trained policy…');
    loadWeights(id, this.getAttribute('base') || BASE).then(function (payload) {
      if (g !== self.gen) return;
      var net = self.nets && self.nets[id];
      if (!net) { self.nets = self.nets || {}; net = self.nets[id] = new M.Net(payload); }
      self.payload = payload;
      self.L = new Player('L', self.modeId, payload, net, self.seed);
      self.R = new Player('R', 'learned', payload, net, self.seed);
      self.setNote('');
      self.resetStrips();
      self.renderTop();
      self.renderAll();
      self.loop(g);
    }, function () {
      if (g === self.gen) self.setNote('The trained policy could not be loaded (weights/' + id + '.js).');
    });
  };

  Demo.prototype.loop = function (g) {
    var self = this, L = this.L, R = this.R;
    function live() { return g === self.gen; }
    function step() {
      if (!live()) return;
      if (L.hist.length) { self.closeStep(L); self.closeStep(R); }                             /* the outcomes of the last step are in */
      L.reveal(); R.reveal();
      L.phase = R.phase = 'arrive';
      self.describeArrive(L); self.describeArrive(R);
      if (!self.quick()) self.renderAll();
      var first = L.info.first;
      self.pause(first ? PACE.first : PACE.arrive, g).then(function () {
        if (!live()) return;
        L.phase = R.phase = 'decide';
        L.info.first = R.info.first = false;
        R.decideAuto();
        if (L.mode !== 'you') L.decideAuto();
        if (first) return;                                                                     /* nothing is stored yet */
        if (L.mode !== 'you') return replay();
        var waiting = new Promise(function (done) { self.userDone = done; });
        replay();
        return waiting.then(function () { R.shown = R.ops.length; });
      }).then(function () {
        if (!live()) return;
        L.phase = R.phase = 'travel';
        L.send(); R.send();
        [L, R].forEach(function (p) {
          var rec = p.hist[p.hist.length - 1];
          p.say = rec.ops.length ? 'The measurement outcomes are crossing the link. One step passes.'
            : rec.t === 0 ? 'After one step the nodes know which attempts succeeded.'
            : 'No operation in this step. One step passes.';
        });
        if (!self.quick()) self.renderAll(); else self.renderQuick();
        return self.pause(PACE.travel, g);
      }).then(step);
    }
    /* show the operations one at a time */
    function replay() {
      var mine = L.mode === 'you', n = mine ? R.ops.length : Math.max(L.ops.length, R.ops.length), i = 0;
      function next() {
        if (!live() || L.phase !== 'decide') return;
        L.shown = mine ? L.ops.length : Math.min(i, L.ops.length);
        R.shown = Math.min(i, R.ops.length);
        self.describeDecide(L); self.describeDecide(R);
        if (!self.quick()) self.renderAll();
        if (i >= n) return self.pause(n ? PACE.after : PACE.idle, g);
        i++;
        return self.pause(PACE.op, g).then(next);
      }
      if (self.quick()) { L.shown = L.ops.length; R.shown = R.ops.length; return Promise.resolve(); }
      return next();
    }
    step();
  };

  /* ---------- sentences ---------- */
  Demo.prototype.who = function (p) {
    return p.mode === 'learned' ? 'The learned policy' : MODES.filter(function (m) { return m.id === p.mode; })[0].name;
  };
  Demo.prototype.describeArrive = function (p) {
    var info = p.info, parts = [], ok = 0, bad = 0, lost = 0, good = 0, low = 0;
    info.results.forEach(function (r) { if (r.ok) ok++; else { bad++; lost += leavesOf(r.tag); } });
    info.delivered.forEach(function (d) { if (d.F >= p.thr) good++; else low++; });
    if (info.first) { p.say = 'The memories are empty. Every memory attempts to generate a pair.'; return; }
    if (ok || bad) parts.push('Outcomes read: ' + (ok ? plural(ok, 'block') + ' succeeded' : '') + (ok && bad ? ', ' : '') + (bad ? plural(bad, 'block') + ' failed (' + plural(lost, 'pair') + ' lost)' : '') + '.');
    if (good) parts.push(plural(good, 'pair') + ' delivered.');
    if (low) parts.push(plural(low, 'pair') + ' came out below ' + p.thr.toFixed(2) + ' and ' + (low === 1 ? 'does' : 'do') + ' not count.');
    parts.push(info.arrivals.length ? plural(info.arrivals.length, 'new pair') + '.' : 'No new pair.');
    p.say = parts.join(' ');
  };
  Demo.prototype.describeDecide = function (p) {
    var n = p.shown, bl = p.blocks();
    if (p.mode === 'you') return this.describeYou(p);
    if (!p.ops.length) { p.say = this.who(p) + ' applies no operation in this step.'; return; }
    if (!n) { p.say = this.who(p) + ' applies ' + plural(p.ops.length, 'operation') + ' in this step.'; return; }
    if (n < p.ops.length) { p.say = this.who(p) + ': operation ' + n + ' of ' + p.ops.length + '.'; return; }
    p.say = this.who(p) + ' applied ' + plural(p.ops.length, 'operation') + ' in ' + plural(bl.length, 'block') + ': ' +
      bl.map(function (b) { return pct(b.p); }).join(', ') + (bl.length > 1 ? ' chances' : ' chance') + ' of success.';
  };
  Demo.prototype.describeYou = function (p) {
    var link = p.link, bl = p.blocks(), thr = p.thr;
    if (p.sel != null && p.hover != null && p.hover !== p.sel && link.canPair(p.sel, p.hover)) {
      var v = link.preview(p.sel, p.hover);
      p.say = 'Memory ' + (p.sel + 1) + ' with memory ' + (p.hover + 1) + ': succeeds ' + pct(v.pBlock) + ', then fidelity ' + f4(v.Fif) + (v.Fif >= thr ? ', enough to deliver.' : ', below ' + thr.toFixed(2) + '.');
    } else if (p.sel != null) {
      p.say = 'Memory ' + (p.sel + 1) + ' is picked. Click a second memory to distill the two pairs.';
    } else if (bl.length) {
      p.say = bl.map(function (b) {
        return 'Memory ' + (b.k + 1) + ': ' + plural(b.ops, 'operation') + ', succeeds ' + pct(b.p) + ', then ' + f4(b.Fif) + (b.Fif >= thr ? ' (delivered)' : '') + '.';
      }).join(' ') + ' Send the outcomes, or keep clicking.';
    } else if (p.before.filter(function (s) { return s.occ; }).length < 2) {
      p.say = 'Your turn. Fewer than two pairs are stored, so wait for the next ones.';
    } else {
      p.say = 'Your turn. Click two memories to distill their pairs, or wait.';
    }
  };

  /* ---------- the visitor ---------- */
  Demo.prototype.onMemory = function (k) {
    var L = this.L;
    if (!L || L.mode !== 'you' || L.phase !== 'decide' || !this.userDone) return;
    var s = L.link.slots[k];
    if (L.sel == null) { if (s.occ) L.sel = k; }
    else if (L.sel === k) L.sel = null;
    else if (L.link.canPair(L.sel, k)) { L.userOp(L.sel, k); L.sel = null; L.hover = null; }
    else if (s.occ) L.sel = k;
    this.renderHalf(L);
  };

  /* ---------- drawing ---------- */
  Demo.prototype.half = function (p) { return this.sh.querySelector('.half[data-side="' + p.side + '"]'); };
  Demo.prototype.setNote = function (t) { this.sh.querySelector('.lead').textContent = t; this.note = t; };

  Demo.prototype.renderTop = function () {
    var P = this.preset, sh = this.sh, pay = this.payload && this.payload.id === P.id ? this.payload : null;
    sh.querySelector('.facts').innerHTML = 'Pairs are generated with fidelity <b>' + P.F0.toFixed(2) + '</b>, and the application needs at least <b>' + P.Fthr.toFixed(2) + '</b>. ' +
      'The nodes close this gap by entanglement distillation (two-to-one, DEJMPS). A policy schedules the operations to maximize the throughput, the pairs delivered per step.' +
      '<span class="more">The other parameters are the same in every setting: each node has eight memories with a coherence time of 200 steps, and the gate error is 0.005. New pairs are Werner states, and stored pairs depolarize.</span>';
    var lr = sh.querySelector('.longrun');
    if (pay) {
      lr.innerHTML = '<div class="lbl">Long-run throughput in this setting, as a share of the upper bound*</div>' +
        '<div class="lr-rows">' +
        [['Nested', 'always waits', pay.paper.nested.rate, ''], ['Optimistic', 'never waits', pay.paper.optimistic.rate, ''], ['Learned policy', '', pay.paper.learned, ' me']].map(function (r) {
          var share = r[2] / pay.paper.bound;
          return '<div class="lr' + r[3] + '"><span class="lr-who"><b>' + r[0] + '</b>' + (r[1] ? '<i>' + r[1] + '</i>' : '') + '</span><span class="lr-b"><i style="width:' + (100 * share).toFixed(1) + '%"></i></span><span class="lr-n">' + Math.round(100 * share) + '%</span></div>';
        }).join('') +
        '<p class="lr-note">The upper bound holds for every policy and may not be attainable.</p></div>' +
        '<p class="lr-why">' + esc(COMPARE) + '</p>';
    }
    var mode = MODES.filter(function (m) { return m.id === this.modeId; }, this)[0];
    sh.querySelector('.half[data-side="L"] .name').textContent = mode.name;
    sh.querySelector('.half[data-side="L"] .sub').textContent = mode.sub;
    sh.querySelector('.foot').innerHTML = (pay ? '<span>* The upper bound of our paper: in the long run, no scheduling policy can deliver more than <b>' + pay.paper.bound.toFixed(2) + '</b> pairs per step in this setting. The bound may not be attainable: possibly no policy delivers that much. The long-run throughput of each policy is computed as in the evaluation of our paper.</span>' : '') +
      '<span>Both sides use the same random numbers for their generation attempts and their operations. The learned policy is the network we trained for this setting in our paper; it runs on this page. <a href="' + PAPER + '" target="_blank" rel="noopener">Read the paper</a></span>';
    this.renderRun();
  };
  Demo.prototype.renderRun = function () {
    var b = this.sh.querySelector('.play'), man = this.manual();
    b.textContent = this.paused ? 'Play' : 'Pause';
    b.setAttribute('aria-label', this.paused ? 'Play' : 'Pause');
    this.sh.querySelector('.ctl').classList.toggle('manual', man);
  };

  Demo.prototype.renderAll = function () {
    if (!this.L) return;
    this.renderHalf(this.L);
    this.renderHalf(this.R);
    this.renderLead();
  };
  Demo.prototype.renderQuick = function () {           /* at Max: the numbers every step, the pictures once per frame */
    var now = root.performance.now();
    if (now - (this.lastDraw || 0) > 40) { this.lastDraw = now; this.renderAll(); }
  };
  Demo.prototype.renderLead = function () {
    var d = this.R.delivered - this.L.delivered, el = this.sh.querySelector('.lead'), left = this.who(this.L), ph = this.L.phase;
    function part(id, text) { return '<span class="ph' + (ph === id ? ' on' : '') + '">' + text + '</span>'; }
    el.innerHTML = '<b>Step ' + this.L.stepNo() + '</b>' + part('arrive', 'outcomes arrive') + part('decide', 'operations') + part('travel', 'outcomes sent') +
      '<span class="gap">' + (d === 0 ? 'Level so far' : d > 0 ? 'The learned policy leads by ' + plural(d, 'pair') : esc(left) + (left === 'You' ? ' lead by ' : ' leads by ') + plural(-d, 'pair')) + '</span>';
  };

  Demo.prototype.renderHalf = function (p) {
    var h = this.half(p), bound = this.payload.paper.bound, rate = p.delivered / p.stepNo();
    h.querySelector('.score .n').textContent = p.delivered;
    h.querySelector('.score .unit').textContent = p.delivered === 1 ? 'pair delivered' : 'pairs delivered';
    h.querySelector('.score .rate').textContent = rate.toFixed(2) + ' per step';
    var span = bound * 1.2;
    h.querySelector('.meter .fill').style.width = (100 * Math.min(1, rate / span)).toFixed(1) + '%';
    h.querySelector('.meter .tick').style.left = (100 * bound / span).toFixed(1) + '%';
    h.querySelector('.meter .cap').textContent = 'upper bound* ' + bound.toFixed(2);
    h.querySelector('.meter .cap').style.left = (100 * bound / span).toFixed(1) + '%';
    h.querySelector('.nodes').innerHTML = this.nodesSVG(p);
    if (p.mode === 'you' && p.phase === 'decide' && this.userDone) this.describeYou(p);
    h.querySelector('.say').textContent = p.say || '';
    this.renderActs(p);
    this.renderStrip(p);
    this.renderTrees(p);
    this.renderTally(p);
  };
  Demo.prototype.renderSay = function (p) {
    if (p.mode === 'you' && p.phase === 'decide' && this.userDone) this.describeYou(p);
    this.half(p).querySelector('.say').textContent = p.say || '';
  };
  Demo.prototype.renderActs = function (p) {
    var el = this.half(p).querySelector('.acts'), html = '';
    if (p.mode === 'you') {
      var mine = p.phase === 'decide' && !!this.userDone;
      if (mine && p.sel != null && p.link.canDiscard(p.sel)) html += '<button type="button" data-act="discard">Discard</button>';
      html += '<button type="button" data-act="undo"' + (mine && p.undo.length ? '' : ' disabled') + '>Undo</button>';
      html += '<button type="button" class="main" data-act="send"' + (mine ? '' : ' disabled') + '>' + (p.phase === 'travel' ? 'Sending…' : p.ops.length ? 'Send outcomes' : 'Wait one step') + '</button>';
    }
    if (el._html !== html) { el.innerHTML = html; el._html = html; }
  };

  /* the two nodes, the memories inside them, the link */
  Demo.prototype.nodesSVG = function (p) {
    var D = p.display(), thr = p.thr, G = NODE, s = '', mine = p.mode === 'you' && p.phase === 'decide' && !!this.userDone;
    var link = p.link, k, last = p.hist[p.hist.length - 1];
    var travel = p.phase === 'travel' && !!last && last.ops.length > 0;                        /* outcomes are on the link */
    s += '<line class="fiber" x1="' + (G.ax + G.R) + '" y1="' + G.y + '" x2="' + (G.bx - G.R) + '" y2="' + G.y + '"/>';
    s += '<text class="link-t" x="250" y="' + (G.y + (travel ? 26 : 17)) + '" text-anchor="middle">link</text>';
    if (travel) {
      s += '<g class="travel"><path class="msg" d="M' + (G.ax + G.R + 8) + ' ' + (G.y - 9) + 'H' + (G.bx - G.R - 10) + '" marker-end="url(#ahr)"/><path class="msg" d="M' + (G.bx - G.R - 8) + ' ' + (G.y + 9) + 'H' + (G.ax + G.R + 10) + '" marker-end="url(#ahr)"/>' +
        '<text class="msg-t" x="250" y="' + (G.y - 17) + '" text-anchor="middle">outcomes</text></g>';
    }
    var tickA = -90 + 360 * gaugeFrac(thr);
    [G.ax, G.bx].forEach(function (cx, ni) {
      s += '<circle class="node" cx="' + cx + '" cy="' + G.y + '" r="' + G.R + '"/>';
      s += '<text class="node-t" x="' + cx + '" y="' + (G.y + G.R + 16) + '" text-anchor="middle">node ' + (ni ? 'B' : 'A') + '</text>';
      s += '<text class="link-t" x="' + cx + '" y="' + (G.y + G.R + 30) + '" text-anchor="middle">' + (ni ? 'does the same as node A' : mine ? 'click two memories' : '') + '</text>';
      s += '<g' + (ni ? ' class="mirror"' : '') + '>';
      for (var i = 0; i < p.shown && p.phase === 'decide'; i++) {                 /* the operations of this step */
        var o = p.ops[i], a = polar(cx, G.y, G.ring - 17, ANG[o.use]), b = polar(cx, G.y, G.ring - 17, ANG[o.keep]);
        var mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, qx = mx + (cx - mx) * 0.6, qy = my + (G.y - my) * 0.6;
        s += '<path class="bond" d="M' + r2(a[0]) + ' ' + r2(a[1]) + 'Q' + r2(qx) + ' ' + r2(qy) + ' ' + r2(b[0]) + ' ' + r2(b[1]) + '"/><circle class="bond-d" cx="' + r2(b[0]) + '" cy="' + r2(b[1]) + '" r="3"/>';
      }
      for (k = 0; k < p.N; k++) {
        var d = D[k], c = polar(cx, G.y, G.ring, ANG[k]), x = c[0], y = c[1], cls = 'mem m-' + d.kind + (d.fresh ? ' fresh' : '') + (d.ok ? ' good' : '') + (d.low ? ' low' : '');
        var can = mine && !ni && link.slots[k].occ, t = polar(x, y, G.g + 3.6, tickA), t0 = polar(x, y, G.g - 2.4, tickA);
        s += '<g class="' + cls + (p.sel === k ? ' sel' : '') + (can ? ' can' : '') + '"' + (ni ? '' : ' data-k="' + k + '"') + (can ? ' tabindex="0" role="button" aria-label="Memory ' + (k + 1) + '"' : '') + '>';
        s += '<circle class="hit" cx="' + r2(x) + '" cy="' + r2(y) + '" r="19"/>';
        s += '<circle class="track" cx="' + r2(x) + '" cy="' + r2(y) + '" r="' + G.g + '"/>';
        if (d.kind === 'kept' || d.kind === 'pend') s += '<path class="arc ghost" d="' + arcPath(x, y, G.g, gaugeFrac(d.kind === 'kept' ? d.Fif : d.F)) + '"/>';
        if (d.kind === 'pair' || d.kind === 'kept' || d.kind === 'used' || d.kind === 'deliv') s += '<path class="arc" d="' + arcPath(x, y, G.g, gaugeFrac(d.F)) + '"/>';
        if (d.kind !== 'gen' && d.kind !== 'fail') s += '<line class="goal" x1="' + r2(t0[0]) + '" y1="' + r2(t0[1]) + '" x2="' + r2(t[0]) + '" y2="' + r2(t[1]) + '"/>';
        s += '<circle class="core" cx="' + r2(x) + '" cy="' + r2(y) + '" r="' + G.m + '"/>';
        if (d.kind === 'fail') s += '<path class="cross" d="M' + r2(x - 4.5) + ' ' + r2(y - 4.5) + 'l9 9M' + r2(x + 4.5) + ' ' + r2(y - 4.5) + 'l-9 9"/>';
        else if (d.kind === 'deliv' && !d.low) s += '<path class="tickmark" d="M' + r2(x - 4.6) + ' ' + r2(y + 0.4) + 'l3.2 3.2l6 -6.6"/>';
        else s += '<text class="idx" x="' + r2(x) + '" y="' + r2(y + 3.4) + '" text-anchor="middle">' + (k + 1) + '</text>';
        var lp = polar(cx, G.y, G.ring + 27, ANG[k]);
        if (ni) { s += '</g>'; continue; }                                                     /* numbers are written in node A only */
        if (p.sel != null && p.sel !== k && mine && link.canPair(p.sel, k)) {                   /* what this partner would give */
          var v = link.preview(p.sel, k);
          s += '<path class="arc hint" d="' + arcPath(x, y, G.g, gaugeFrac(v.Fif)) + '"/><text class="hint-t" x="' + r2(lp[0]) + '" y="' + r2(lp[1] + 3.2) + '" text-anchor="middle">' + pct(v.pBlock) + '</text>';
        } else if (d.kind === 'kept' || d.kind === 'pend') {
          s += '<text class="chance" x="' + r2(lp[0]) + '" y="' + r2(lp[1] + 3.2) + '" text-anchor="middle">' + pct(d.p) + '</text>';
        }
        s += '</g>';
      }
      s += '</g>';
    });
    return s;
  };

  /* the circuit: one group per step, the last one being built as the operations are applied */
  Demo.prototype.resetStrips = function () {
    var self = this, C = CIRC;
    ['L', 'R'].forEach(function (side) {
      var h = self.sh.querySelector('.half[data-side="' + side + '"]'), strip = h.querySelector('.strip');
      strip.innerHTML = '';
      strip._n = 0; strip._first = 0; strip._w = 0;
      h.querySelector('.scroll')._free = false;
      var rows = '', k;
      rows += '<text class="lane-t" transform="translate(8,' + (C.top + C.lane / 2) + ') rotate(-90)" text-anchor="middle">node A</text><text class="lane-t" transform="translate(8,' + (C.top + C.lane + C.gap + C.lane / 2) + ') rotate(-90)" text-anchor="middle">node B</text>';
      for (k = 0; k < 8; k++) rows += '<text class="row-t" x="24" y="' + (self.yA(k) + 3) + '" text-anchor="middle">' + (k + 1) + '</text><text class="row-t" x="24" y="' + (self.yB(k) + 3) + '" text-anchor="middle">' + (k + 1) + '</text>';
      h.querySelector('.rows').innerHTML = rows;
    });
  };
  Demo.prototype.yA = function (k) { return CIRC.top + CIRC.pad + k * CIRC.ws; };
  Demo.prototype.yB = function (k) { return CIRC.top + CIRC.lane + CIRC.gap + CIRC.pad + (7 - k) * CIRC.ws; };

  Demo.prototype.stepSVG = function (p, rec, live, veil) {     /* veil: sent, outcomes not yet known */
    var C = CIRC, self = this, s = '', k, ncol = 0, shown = live ? p.shown : rec.ops.length;
    var ops = rec.ops.slice(0, shown);
    ops.forEach(function (o) { ncol = Math.max(ncol, o.col + 1); });
    var gs = Math.min(13, 66 / Math.max(1, ncol)), gx = function (c) { return 32 + c * gs; }, xm = 32 + ncol * gs + 7, xnow = Math.min(C.sw - 6, xm + 6);
    var yBot = C.top + 2 * C.lane + C.gap;
    s += '<rect class="lane" x="0" y="' + C.top + '" width="' + C.sw + '" height="' + C.lane + '"/><rect class="lane" x="0" y="' + (C.top + C.lane + C.gap) + '" width="' + C.sw + '" height="' + C.lane + '"/>';
    s += '<line class="guide" x1="0" y1="' + (C.top - 2) + '" x2="0" y2="' + (yBot + 3) + '"/>';
    s += '<text class="step-t" x="' + C.sw / 2 + '" y="' + (yBot + 14) + '" text-anchor="middle">step ' + (rec.t + 1) + '</text>';
    var usedAt = {};
    ops.forEach(function (o) { usedAt[o.use] = gx(o.col); });
    for (k = 0; k < p.N; k++) {
      if (!rec.start[k]) continue;
      var x1 = rec.start[k] === 'new' ? 12 : 0, x2, mark = '';
      var gone = rec.discards.indexOf(k) >= 0;
      if (gone) { x2 = 22; }
      else if (usedAt[k] != null) x2 = rec.sent ? xm - 6 : usedAt[k] + 8;
      else x2 = rec.sent ? (!veil && rec.outcome[k] === 'fail' ? C.sw - 6 : C.sw) : xnow;
      [self.yA(k), self.yB(k)].forEach(function (y) {
        s += '<line class="wire" x1="' + x1 + '" y1="' + y + '" x2="' + r2(x2) + '" y2="' + y + '"/>';
        if (rec.start[k] === 'new') s += '<circle class="bub" cx="12" cy="' + y + '" r="3.6"/>';
        if (gone) s += '<path class="gone" d="M' + 19.5 + ' ' + (y - 2.5) + 'l5 5M' + 24.5 + ' ' + (y - 2.5) + 'l-5 5"/>';
        if (rec.sent && usedAt[k] != null) s += '<rect class="mtr" x="' + r2(xm - 6) + '" y="' + (y - 4.5) + '" width="12" height="9" rx="1.4"/><path class="mtr-a" d="M' + r2(xm - 3.4) + ' ' + (y + 2) + 'A3.6 3.6 0 0 1 ' + r2(xm + 3.4) + ' ' + (y + 2) + 'M' + r2(xm) + ' ' + (y + 2.2) + 'L' + r2(xm + 2.8) + ' ' + (y - 2.8) + '"/>';
        var oc = veil ? null : rec.outcome[k];
        if (oc === 'fail') s += '<path class="cross" d="M' + (C.sw - 9) + ' ' + (y - 3) + 'l6 6M' + (C.sw - 3) + ' ' + (y - 3) + 'l-6 6"/>';
        else if (oc === 'deliv') s += '<circle class="done" cx="' + (C.sw - 4.5) + '" cy="' + y + '" r="3.8"/>';
        else if (oc === 'low') s += '<circle class="done low" cx="' + (C.sw - 4.5) + '" cy="' + y + '" r="3.2"/>';
      });
    }
    ops.forEach(function (o) {
      var x = gx(o.col);
      [[self.yA(o.keep), self.yA(o.use)], [self.yB(o.keep), self.yB(o.use)]].forEach(function (y) {
        s += '<line class="gate" x1="' + r2(x) + '" y1="' + y[0] + '" x2="' + r2(x) + '" y2="' + y[1] + '"/><circle class="gate-d" cx="' + r2(x) + '" cy="' + y[0] + '" r="2.3"/>' +
          '<circle class="gate-t" cx="' + r2(x) + '" cy="' + y[1] + '" r="4"/><path class="gate" d="M' + r2(x - 4) + ' ' + y[1] + 'H' + r2(x + 4) + 'M' + r2(x) + ' ' + (y[1] - 4) + 'V' + (y[1] + 4) + '"/>';
      });
    });
    if (rec.sent && rec.ops.length) {
      var xs = xm + 10, xe = C.sw - 3, ya = C.top + C.lane, yb = C.top + C.lane + C.gap;
      s += '<path class="msg" d="M' + r2(xs) + ' ' + ya + 'L' + xe + ' ' + yb + '" marker-end="url(#ahr)"/><path class="msg" d="M' + r2(xs) + ' ' + yb + 'L' + xe + ' ' + ya + '" marker-end="url(#ahr)"/>';
    }
    return s;
  };

  Demo.prototype.closeStep = function (p) {            /* the step that was just sent becomes part of the history */
    var strip = this.half(p).querySelector('.strip'), rec = p.hist[p.hist.length - 1];
    var g = strip.querySelector('g.now');
    if (g) { g.setAttribute('class', 'st'); g.innerHTML = this.stepSVG(p, rec, false); }
    else this.appendStep(strip, this.stepSVG(p, rec, false), 'st');
    if (strip._n - strip._first > 240) {                /* keep the last steps only */
      var drop = 120, sc = this.half(p).querySelector('.scroll'), kids = strip.querySelectorAll('g.st');
      for (var i = 0; i < drop; i++) strip.removeChild(kids[i]);
      strip._first += drop;
      Array.prototype.forEach.call(strip.children, function (c, j) { c.setAttribute('transform', 'translate(' + j * CIRC.sw + ',0)'); });
      sc.scrollLeft = Math.max(0, sc.scrollLeft - drop * CIRC.sw);
    }
  };
  Demo.prototype.appendStep = function (strip, html, cls) {
    var g = doc.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', cls);
    g.setAttribute('transform', 'translate(' + (strip._n - strip._first) * CIRC.sw + ',0)');
    g.innerHTML = html;
    strip.appendChild(g);
    strip._n++;
    return g;
  };
  Demo.prototype.renderStrip = function (p) {
    var h = this.half(p), strip = h.querySelector('.strip'), sc = h.querySelector('.scroll');
    var sent = p.phase === 'travel' && p.hist.length > 0;       /* while the outcomes travel, the step that sent them stays open */
    var g = strip.querySelector('g.now'), html = sent ? this.stepSVG(p, p.hist[p.hist.length - 1], false, true) : this.stepSVG(p, p.rec, true, false);
    if (g) g.innerHTML = html; else this.appendStep(strip, html, 'now');
    var w = (strip._n - strip._first) * CIRC.sw + 14;
    if (strip._w !== w) { strip.setAttribute('width', w); strip.setAttribute('viewBox', '0 0 ' + w + ' ' + CIRC.h); strip._w = w; }
    if (!sc._free) sc.scrollLeft = sc.scrollWidth;
  };

  Demo.prototype.renderTrees = function (p) {
    var el = this.half(p).querySelector('.trees'), list = p.treesNow(), html = '';
    list.forEach(function (t) {
      html += '<div class="tcard' + (t.pend ? ' pend' : '') + '"><div class="g">' + treeSVG(t.tag, false, 58, 40) + '</div><div class="c"><b>memory ' + (t.k + 1) + '</b><span>' +
        (t.pend ? pct(t.p) + ' · ' + f4(t.F) : f4(t.F)) + '</span></div></div>';
    });
    if (!html) html = '<p class="none">No distilled pair is stored.</p>';
    if (el._html !== html) { el.innerHTML = html; el._html = html; }
  };

  Demo.prototype.renderTally = function (p) {
    var h = this.half(p), el = h.querySelector('.tally'), rows = [], total = p.delivered, html = '';
    for (var tag in p.tally) rows.push([tag, p.tally[tag]]);
    rows.sort(function (a, b) { return b[1] - a[1] || (a[0] < b[0] ? -1 : 1); });
    h.querySelector('.kinds').textContent = rows.length ? plural(rows.length, 'kind') : '';
    rows.slice(0, 5).forEach(function (r) {
      var share = r[1] / total;
      html += '<div class="trow"><div class="g">' + treeSVG(r[0], true, 58, 34) + '</div><code>' + strHTML(r[0]) + '</code><div class="b"><i style="width:' + (100 * share).toFixed(1) + '%"></i></div><span class="s">' + pct(share) + '</span></div>';
    });
    if (rows.length > 5) html += '<p class="none">and ' + plural(rows.length - 5, 'other kind') + '</p>';
    if (p.low) html += '<p class="none">' + plural(p.low, 'pair') + ' delivered below ' + p.thr.toFixed(2) + ', not counted.</p>';
    if (!html) html = '<p class="none">Nothing delivered yet.</p>';
    if (el._html !== html) { el.innerHTML = html; el._html = html; }
  };

  root.customElements.define('distillation-demo', Demo);
})();


})(typeof window !== 'undefined' ? window : globalThis);
