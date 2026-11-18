// A one-block transformer, small enough to run and train in a browser tab.
// It keeps every intermediate result so lessons can draw the real numbers.

import {
  Mat, add, addRow, backward, causalSoftmax, concatCols, crossEntropy,
  gather, matmul, relu, scale, softmaxRow, transpose,
} from './tensor';
import { gaussian } from './data';

export interface Config {
  vocab: number; // how many different tokens exist
  ctx: number; // longest sequence the model can read
  d: number; // size of each token's vector
  heads: number; // attention heads
  hidden: number; // width of the MLP layer
}

export interface Trace {
  tokEmb: Mat; posEmb: Mat; x: Mat;
  q: Mat[]; k: Mat[]; v: Mat[]; scores: Mat[]; att: Mat[]; mixed: Mat[];
  attOut: Mat; x1: Mat; mlpOut: Mat; x2: Mat; logits: Mat;
}

export interface Weights { config: Config; params: Record<string, number[]> }

export class GPT {
  p: Record<string, Mat> = {};

  constructor(public cfg: Config, rand: () => number) {
    const { vocab, ctx, d, heads, hidden } = cfg;
    const dh = d / heads;
    const init = (r: number, c: number, std: number) => {
      const m = new Mat(r, c);
      for (let i = 0; i < m.d.length; i++) m.d[i] = gaussian(rand) * std;
      return m;
    };
    this.p.tok = init(vocab, d, 0.3);
    this.p.pos = init(ctx, d, 0.3);
    for (let h = 0; h < heads; h++) {
      this.p['q' + h] = init(d, dh, 1 / Math.sqrt(d));
      this.p['k' + h] = init(d, dh, 1 / Math.sqrt(d));
      this.p['v' + h] = init(d, dh, 1 / Math.sqrt(d));
    }
    this.p.o = init(d, d, 1 / Math.sqrt(d));
    this.p.w1 = init(d, hidden, 1 / Math.sqrt(d));
    this.p.b1 = new Mat(1, hidden);
    this.p.w2 = init(hidden, d, 1 / Math.sqrt(hidden));
    this.p.b2 = new Mat(1, d);
    this.p.u = init(d, vocab, 1 / Math.sqrt(d));
    this.p.bu = new Mat(1, vocab);
  }

  params(): Mat[] { return Object.values(this.p); }
  paramCount(): number { return this.params().reduce((s, m) => s + m.d.length, 0); }
  zeroGrad(): void { for (const m of this.params()) m.g.fill(0); }

  forward(ids: number[]): Trace {
    const P = this.p;
    const dh = this.cfg.d / this.cfg.heads;
    const tokEmb = gather(P.tok, ids);
    const posEmb = gather(P.pos, ids.map((_, i) => i));
    const x = add(tokEmb, posEmb);

    const q: Mat[] = [], k: Mat[] = [], v: Mat[] = [], scores: Mat[] = [], att: Mat[] = [], mixed: Mat[] = [];
    for (let h = 0; h < this.cfg.heads; h++) {
      q.push(matmul(x, P['q' + h]));
      k.push(matmul(x, P['k' + h]));
      v.push(matmul(x, P['v' + h]));
      scores.push(scale(matmul(q[h], transpose(k[h])), 1 / Math.sqrt(dh)));
      att.push(causalSoftmax(scores[h]));
      mixed.push(matmul(att[h], v[h]));
    }
    const attOut = matmul(mixed.length === 1 ? mixed[0] : concatCols(mixed), P.o);
    const x1 = add(x, attOut);

    const mlpOut = addRow(matmul(relu(addRow(matmul(x1, P.w1), P.b1)), P.w2), P.b2);
    const x2 = add(x1, mlpOut);
    const logits = addRow(matmul(x2, P.u), P.bu);
    return { tokEmb, posEmb, x, q, k, v, scores, att, mixed, attOut, x1, mlpOut, x2, logits };
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

  /** Write one name, a letter at a time. */
  sample(rand: () => number, temperature = 1): string {
    const ids = [0];
    while (ids.length < this.cfg.ctx) {
      const p = this.next(ids, temperature);
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

export const DEFAULT_CONFIG: Config = { vocab: 27, ctx: 8, d: 16, heads: 1, hidden: 32 };
