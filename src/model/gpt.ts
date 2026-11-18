// A small transformer, small enough to run and train in a browser tab.
// It keeps every intermediate result so lessons can draw the real numbers.
//
// Weight names: layer 0 uses q0, k0, v0, o, w1, b1, w2, b2; layer n > 0 adds
// the suffix `_n` (q0_1, o_1, …). Embeddings are tok and pos, the output is u, bu.

import {
  Mat, add, addRow, backward, causalSoftmax, concatCols, crossEntropy,
  gather, layerNorm, matmul, relu, scale, softmaxRow, transpose,
} from './tensor';
import { gaussian } from './data';

export interface Config {
  vocab: number; // how many different tokens exist
  ctx: number; // longest sequence the model can read
  d: number; // size of each token's vector
  heads: number; // attention heads per layer
  hidden: number; // width of the MLP layer
  layers?: number; // transformer blocks stacked on top of each other (default 1)
  norm?: boolean; // layer normalization before attention, MLP and output (default off)
  noPos?: boolean; // leave out position embeddings (for experiments only)
}

export interface BlockTrace {
  ln1?: Mat; q: Mat[]; k: Mat[]; v: Mat[]; scores: Mat[]; att: Mat[]; mixed: Mat[];
  attOut: Mat; x1: Mat; ln2?: Mat; hidden: Mat; mlpOut: Mat; x2: Mat;
}

/** Layer 0's fields are copied to the top level; `blocks` holds every layer. */
export interface Trace extends BlockTrace {
  tokEmb: Mat; posEmb: Mat | null; x: Mat; blocks: BlockTrace[]; final: Mat; logits: Mat;
}

export interface Weights { config: Config; params: Record<string, number[]> }

const sfx = (l: number) => (l === 0 ? '' : '_' + l);

export class GPT {
  p: Record<string, Mat> = {};
  /** LoRA adapters: for a weight W, the model uses W + A·B × (alpha / rank). */
  lora: Record<string, { A: Mat; B: Mat }> = {};
  loraScale = 1;
  /** When true, only the LoRA adapters are trained. */
  frozen = false;

  constructor(public cfg: Config, rand: () => number) {
    const { vocab, ctx, d, heads, hidden } = cfg;
    const dh = d / heads;
    const init = (r: number, c: number, std: number) => {
      const m = new Mat(r, c);
      for (let i = 0; i < m.d.length; i++) m.d[i] = gaussian(rand) * std;
      return m;
    };
    const ones = (c: number) => { const m = new Mat(1, c); m.d.fill(1); return m; };
    this.p.tok = init(vocab, d, 0.3);
    if (!cfg.noPos) this.p.pos = init(ctx, d, 0.3);
    for (let l = 0; l < (cfg.layers ?? 1); l++) {
      const s = sfx(l);
      if (cfg.norm) { this.p['g1' + s] = ones(d); this.p['c1' + s] = new Mat(1, d); }
      for (let h = 0; h < heads; h++) {
        this.p['q' + h + s] = init(d, dh, 1 / Math.sqrt(d));
        this.p['k' + h + s] = init(d, dh, 1 / Math.sqrt(d));
        this.p['v' + h + s] = init(d, dh, 1 / Math.sqrt(d));
      }
      this.p['o' + s] = init(d, d, 1 / Math.sqrt(d));
      if (cfg.norm) { this.p['g2' + s] = ones(d); this.p['c2' + s] = new Mat(1, d); }
      this.p['w1' + s] = init(d, hidden, 1 / Math.sqrt(d));
      this.p['b1' + s] = new Mat(1, hidden);
      this.p['w2' + s] = init(hidden, d, 1 / Math.sqrt(hidden));
      this.p['b2' + s] = new Mat(1, d);
    }
    if (cfg.norm) { this.p.gf = ones(d); this.p.cf = new Mat(1, d); }
    this.p.u = init(d, vocab, 1 / Math.sqrt(d));
    this.p.bu = new Mat(1, vocab);
  }

  /** The weights that training updates. */
  params(): Mat[] {
    const adapters = Object.values(this.lora).flatMap((x) => [x.A, x.B]);
    return this.frozen ? adapters : [...Object.values(this.p), ...adapters];
  }
  paramCount(): number { return Object.values(this.p).reduce((s, m) => s + m.d.length, 0); }
  loraCount(): number { return Object.values(this.lora).reduce((s, x) => s + x.A.d.length + x.B.d.length, 0); }
  zeroGrad(): void { for (const m of [...Object.values(this.p), ...Object.values(this.lora).flatMap((x) => [x.A, x.B])]) m.g.fill(0); }

  /** Add LoRA adapters of the given rank to the named weights, and freeze everything else. */
  addLoRA(names: string[], rank: number, rand: () => number, alpha = rank): void {
    for (const n of names) {
      const W = this.p[n];
      const A = new Mat(W.r, rank);
      for (let i = 0; i < A.d.length; i++) A.d[i] = gaussian(rand) / Math.sqrt(W.r);
      this.lora[n] = { A, B: new Mat(rank, W.c) }; // B starts at zero, so the model is unchanged at first
    }
    this.loraScale = alpha / rank;
    this.frozen = true;
  }

  /** A weight as the model uses it (with any LoRA adapter added). */
  private w(name: string): Mat {
    const ad = this.lora[name];
    return ad ? add(this.p[name], scale(matmul(ad.A, ad.B), this.loraScale)) : this.p[name];
  }

  forward(ids: number[]): Trace {
    const P = this.p;
    const { heads, d } = this.cfg;
    const dh = d / heads;
    const tokEmb = gather(P.tok, ids);
    const posEmb = this.cfg.noPos ? null : gather(P.pos, ids.map((_, i) => i));
    const x = posEmb ? add(tokEmb, posEmb) : tokEmb;

    const blocks: BlockTrace[] = [];
    let h = x;
    for (let l = 0; l < (this.cfg.layers ?? 1); l++) {
      const s = sfx(l);
      const ln1 = this.cfg.norm ? layerNorm(h, P['g1' + s], P['c1' + s]) : undefined;
      const inA = ln1 ?? h;
      const q: Mat[] = [], k: Mat[] = [], v: Mat[] = [], scores: Mat[] = [], att: Mat[] = [], mixed: Mat[] = [];
      for (let hd = 0; hd < heads; hd++) {
        q.push(matmul(inA, this.w('q' + hd + s)));
        k.push(matmul(inA, this.w('k' + hd + s)));
        v.push(matmul(inA, this.w('v' + hd + s)));
        scores.push(scale(matmul(q[hd], transpose(k[hd])), 1 / Math.sqrt(dh)));
        att.push(causalSoftmax(scores[hd]));
        mixed.push(matmul(att[hd], v[hd]));
      }
      const attOut = matmul(mixed.length === 1 ? mixed[0] : concatCols(mixed), this.w('o' + s));
      const x1 = add(h, attOut);
      const ln2 = this.cfg.norm ? layerNorm(x1, P['g2' + s], P['c2' + s]) : undefined;
      const hidden = relu(addRow(matmul(ln2 ?? x1, this.w('w1' + s)), P['b1' + s]));
      const mlpOut = addRow(matmul(hidden, this.w('w2' + s)), P['b2' + s]);
      const x2 = add(x1, mlpOut);
      blocks.push({ ln1, q, k, v, scores, att, mixed, attOut, x1, ln2, hidden, mlpOut, x2 });
      h = x2;
    }
    const final = this.cfg.norm ? layerNorm(h, P.gf, P.cf) : h;
    const logits = addRow(matmul(final, this.w('u')), P.bu);
    return { ...blocks[0], x2: h, tokEmb, posEmb, x, blocks, final, logits };
  }

  /** Probabilities for the token that follows the last one in `ids`. */
  next(ids: number[], temperature = 1): number[] {
    const { logits } = this.forward(ids);
    return softmaxRow(logits.row(ids.length - 1), temperature);
  }

  /** Average loss over a batch; also fills in gradients when `train` is true. */
  loss(batch: { x: number[]; y: number[] }[], train = false): number {
    if (train) this.zeroGrad();
    let total = 0;
    for (const ex of batch) {
      const l = crossEntropy(this.forward(ex.x).logits, ex.y);
      total += l.d[0];
      if (train) backward(l, 1 / batch.length);
    }
    return total / batch.length;
  }

  /** Length of the whole gradient, treated as one long vector. */
  gradNorm(): number {
    let s = 0;
    for (const m of this.params()) for (let i = 0; i < m.g.length; i++) s += m.g[i] * m.g[i];
    return Math.sqrt(s);
  }

  /** Write one name, a letter at a time. `topK` keeps only the k likeliest letters. */
  sample(rand: () => number, temperature = 1, topK = 0): string {
    const ids = [0];
    while (ids.length < this.cfg.ctx) {
      let p = this.next(ids, temperature);
      if (topK > 0) {
        const cut = [...p].sort((a, b) => b - a)[Math.min(topK, p.length) - 1];
        p = p.map((v) => (v >= cut ? v : 0));
        const z = p.reduce((a, b) => a + b, 0);
        p = p.map((v) => v / z);
      }
      let r = rand(), id = 0;
      while (id < p.length - 1 && (r -= p[id]) > 0) id++;
      if (id === 0) break;
      ids.push(id);
    }
    return ids.slice(1).map((i) => '.abcdefghijklmnopqrstuvwxyz'[i]).join('');
  }

  toJSON(): Weights {
    const params: Record<string, number[]> = {};
    for (const [name, m] of Object.entries(this.p)) params[name] = Array.from(m.d, (v) => +v.toFixed(4));
    return { config: this.cfg, params };
  }

  static fromJSON(w: Weights): GPT {
    const m = new GPT(w.config, () => 0.5);
    for (const [name, vals] of Object.entries(w.params)) m.p[name].d.set(vals);
    return m;
  }

  clone(): GPT { return GPT.fromJSON(this.toJSON()); }
}

/** Plain gradient descent: move each weight a little against its gradient. */
export function sgdStep(model: GPT, lr: number): void {
  for (const m of model.params()) for (let i = 0; i < m.d.length; i++) m.d[i] -= lr * m.g[i];
}

/** Adam: gradient descent with a running average and a per-weight step size. */
export class Adam {
  private m = new Map<Mat, Float64Array>();
  private v = new Map<Mat, Float64Array>();
  private t = 0;
  constructor(private model: GPT, public lr = 0.01) {}
  step(): void {
    this.t++;
    for (const p of this.model.params()) {
      if (!this.m.has(p)) { this.m.set(p, new Float64Array(p.d.length)); this.v.set(p, new Float64Array(p.d.length)); }
      const m = this.m.get(p)!, v = this.v.get(p)!;
      for (let i = 0; i < p.d.length; i++) {
        m[i] = 0.9 * m[i] + 0.1 * p.g[i];
        v[i] = 0.999 * v[i] + 0.001 * p.g[i] * p.g[i];
        const mh = m[i] / (1 - 0.9 ** this.t), vh = v[i] / (1 - 0.999 ** this.t);
        p.d[i] -= (this.lr * mh) / (Math.sqrt(vh) + 1e-8);
      }
    }
  }
}

/** Learning rate with linear warmup, then a cosine decay down to 10% of the peak. */
export function warmupCosine(step: number, total: number, peak: number, warmup: number): number {
  if (step < warmup) return (peak * (step + 1)) / warmup;
  const t = (step - warmup) / Math.max(1, total - warmup);
  return peak * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, t))));
}

export const DEFAULT_CONFIG: Config = { vocab: 27, ctx: 8, d: 16, heads: 1, hidden: 32 };
