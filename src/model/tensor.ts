// A very small matrix library with automatic differentiation.
// Every operation remembers how it was made, so `backward` can walk the
// chain in reverse and fill in `.g` (the gradient) for every matrix.

export class Mat {
  d: Float64Array; // values, row by row
  g: Float64Array; // gradient of the loss with respect to each value
  prev: Mat[] = [];
  back: (() => void) | null = null;

  constructor(public r: number, public c: number, d?: Float64Array) {
    this.d = d ?? new Float64Array(r * c);
    this.g = new Float64Array(r * c);
  }

  row(i: number): number[] {
    return Array.from(this.d.subarray(i * this.c, (i + 1) * this.c));
  }
}

function node(r: number, c: number, prev: Mat[]): Mat {
  const o = new Mat(r, c);
  o.prev = prev;
  return o;
}

export function matmul(a: Mat, b: Mat): Mat {
  if (a.c !== b.r) throw new Error(`matmul shape ${a.r}x${a.c} · ${b.r}x${b.c}`);
  const o = node(a.r, b.c, [a, b]);
  for (let i = 0; i < a.r; i++)
    for (let k = 0; k < a.c; k++) {
      const v = a.d[i * a.c + k];
      for (let j = 0; j < b.c; j++) o.d[i * b.c + j] += v * b.d[k * b.c + j];
    }
  o.back = () => {
    for (let i = 0; i < a.r; i++)
      for (let k = 0; k < a.c; k++) {
        let s = 0;
        const v = a.d[i * a.c + k];
        for (let j = 0; j < b.c; j++) {
          const og = o.g[i * b.c + j];
          s += og * b.d[k * b.c + j];
          b.g[k * b.c + j] += v * og;
        }
        a.g[i * a.c + k] += s;
      }
  };
  return o;
}

export function add(a: Mat, b: Mat): Mat {
  if (a.r !== b.r || a.c !== b.c) throw new Error('add shape');
  const o = node(a.r, a.c, [a, b]);
  for (let i = 0; i < o.d.length; i++) o.d[i] = a.d[i] + b.d[i];
  o.back = () => {
    for (let i = 0; i < o.d.length; i++) {
      a.g[i] += o.g[i];
      b.g[i] += o.g[i];
    }
  };
  return o;
}

/** Add a single row (a bias) to every row of `a`. */
export function addRow(a: Mat, b: Mat): Mat {
  if (b.r !== 1 || b.c !== a.c) throw new Error('addRow shape');
  const o = node(a.r, a.c, [a, b]);
  for (let i = 0; i < a.r; i++) for (let j = 0; j < a.c; j++) o.d[i * a.c + j] = a.d[i * a.c + j] + b.d[j];
  o.back = () => {
    for (let i = 0; i < a.r; i++)
      for (let j = 0; j < a.c; j++) {
        a.g[i * a.c + j] += o.g[i * a.c + j];
        b.g[j] += o.g[i * a.c + j];
      }
  };
  return o;
}

export function scale(a: Mat, s: number): Mat {
  const o = node(a.r, a.c, [a]);
  for (let i = 0; i < o.d.length; i++) o.d[i] = a.d[i] * s;
  o.back = () => {
    for (let i = 0; i < o.d.length; i++) a.g[i] += o.g[i] * s;
  };
  return o;
}

export function relu(a: Mat): Mat {
  const o = node(a.r, a.c, [a]);
  for (let i = 0; i < o.d.length; i++) o.d[i] = a.d[i] > 0 ? a.d[i] : 0;
  o.back = () => {
    for (let i = 0; i < o.d.length; i++) if (a.d[i] > 0) a.g[i] += o.g[i];
  };
  return o;
}

export function transpose(a: Mat): Mat {
  const o = node(a.c, a.r, [a]);
  for (let i = 0; i < a.r; i++) for (let j = 0; j < a.c; j++) o.d[j * a.r + i] = a.d[i * a.c + j];
  o.back = () => {
    for (let i = 0; i < a.r; i++) for (let j = 0; j < a.c; j++) a.g[i * a.c + j] += o.g[j * a.r + i];
  };
  return o;
}

/** Pick rows out of a table: row i of the result is row ids[i] of `table`. */
export function gather(table: Mat, ids: number[]): Mat {
  const o = node(ids.length, table.c, [table]);
  ids.forEach((id, i) => o.d.set(table.d.subarray(id * table.c, (id + 1) * table.c), i * table.c));
  o.back = () => {
    ids.forEach((id, i) => {
      for (let j = 0; j < table.c; j++) table.g[id * table.c + j] += o.g[i * table.c + j];
    });
  };
  return o;
}

/**
 * Softmax over each row of a square score matrix, where row i may only
 * use columns 0..i (a token cannot look at tokens that come after it).
 */
export function causalSoftmax(a: Mat): Mat {
  const n = a.r;
  const o = node(n, n, [a]);
  for (let i = 0; i < n; i++) {
    let m = -Infinity;
    for (let j = 0; j <= i; j++) m = Math.max(m, a.d[i * n + j]);
    let z = 0;
    for (let j = 0; j <= i; j++) z += o.d[i * n + j] = Math.exp(a.d[i * n + j] - m);
    for (let j = 0; j <= i; j++) o.d[i * n + j] /= z;
  }
  o.back = () => {
    for (let i = 0; i < n; i++) {
      let dot = 0;
      for (let j = 0; j <= i; j++) dot += o.g[i * n + j] * o.d[i * n + j];
      for (let j = 0; j <= i; j++) a.g[i * n + j] += o.d[i * n + j] * (o.g[i * n + j] - dot);
    }
  };
  return o;
}

export function concatCols(parts: Mat[]): Mat {
  const r = parts[0].r;
  const c = parts.reduce((s, p) => s + p.c, 0);
  const o = node(r, c, parts);
  let off = 0;
  const offs = parts.map((p) => {
    const at = off;
    for (let i = 0; i < r; i++) for (let j = 0; j < p.c; j++) o.d[i * c + at + j] = p.d[i * p.c + j];
    off += p.c;
    return at;
  });
  o.back = () => {
    parts.forEach((p, n) => {
      for (let i = 0; i < r; i++) for (let j = 0; j < p.c; j++) p.g[i * p.c + j] += o.g[i * c + offs[n] + j];
    });
  };
  return o;
}

export function softmaxRow(logits: number[], temperature = 1): number[] {
  const m = Math.max(...logits);
  const e = logits.map((v) => Math.exp((v - m) / temperature));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

/**
 * Cross-entropy loss: for each row, −ln(probability given to the correct
 * answer), averaged over the rows. Returns a 1×1 matrix.
 */
export function crossEntropy(logits: Mat, targets: number[]): Mat {
  const o = node(1, 1, [logits]);
  const probs: number[][] = [];
  for (let i = 0; i < logits.r; i++) {
    const p = softmaxRow(logits.row(i));
    probs.push(p);
    o.d[0] -= Math.log(p[targets[i]]) / logits.r;
  }
  o.back = () => {
    for (let i = 0; i < logits.r; i++)
      for (let j = 0; j < logits.c; j++)
        logits.g[i * logits.c + j] += (o.g[0] * (probs[i][j] - (j === targets[i] ? 1 : 0))) / logits.r;
  };
  return o;
}

/** Fill in `.g` for everything that `loss` depends on. */
export function backward(loss: Mat, seed = 1): void {
  const order: Mat[] = [];
  const seen = new Set<Mat>();
  const visit = (m: Mat) => {
    if (seen.has(m)) return;
    seen.add(m);
    for (const p of m.prev) visit(p);
    order.push(m);
  };
  visit(loss);
  loss.g[0] = seed;
  for (let i = order.length - 1; i >= 0; i--) order[i].back?.();
}

/**
 * Layer normalization of each row: subtract the row's mean, divide by its
 * standard deviation, then scale by `gain` and shift by `bias` (both 1×c).
 */
export function layerNorm(a: Mat, gain: Mat, bias: Mat, eps = 1e-5): Mat {
  const { r, c } = a;
  const o = node(r, c, [a, gain, bias]);
  const xhat = new Float64Array(r * c);
  const inv = new Float64Array(r);
  for (let i = 0; i < r; i++) {
    let mean = 0;
    for (let j = 0; j < c; j++) mean += a.d[i * c + j];
    mean /= c;
    let v = 0;
    for (let j = 0; j < c; j++) v += (a.d[i * c + j] - mean) ** 2;
    inv[i] = 1 / Math.sqrt(v / c + eps);
    for (let j = 0; j < c; j++) {
      xhat[i * c + j] = (a.d[i * c + j] - mean) * inv[i];
      o.d[i * c + j] = xhat[i * c + j] * gain.d[j] + bias.d[j];
    }
  }
  o.back = () => {
    for (let i = 0; i < r; i++) {
      let sg = 0, sgx = 0;
      for (let j = 0; j < c; j++) {
        const g = o.g[i * c + j];
        gain.g[j] += g * xhat[i * c + j];
        bias.g[j] += g;
        const dx = g * gain.d[j];
        sg += dx;
        sgx += dx * xhat[i * c + j];
      }
      for (let j = 0; j < c; j++) {
        const dx = o.g[i * c + j] * gain.d[j];
        a.g[i * c + j] += (inv[i] / c) * (c * dx - sg - xhat[i * c + j] * sgx);
      }
    }
  };
  return o;
}

/** Softmax over each whole row (no mask). Used by the mixture-of-experts router. */
export function softmaxRows(a: Mat): Mat {
  const { r, c } = a;
  const o = node(r, c, [a]);
  for (let i = 0; i < r; i++) {
    let m = -Infinity;
    for (let j = 0; j < c; j++) m = Math.max(m, a.d[i * c + j]);
    let z = 0;
    for (let j = 0; j < c; j++) z += o.d[i * c + j] = Math.exp(a.d[i * c + j] - m);
    for (let j = 0; j < c; j++) o.d[i * c + j] /= z;
  }
  o.back = () => {
    for (let i = 0; i < r; i++) {
      let dot = 0;
      for (let j = 0; j < c; j++) dot += o.g[i * c + j] * o.d[i * c + j];
      for (let j = 0; j < c; j++) a.g[i * c + j] += o.d[i * c + j] * (o.g[i * c + j] - dot);
    }
  };
  return o;
}

/**
 * Multiply each row i of `a` by w[i][col] × mask[i] (mask is a fixed 0/1 list).
 * Used to weight one expert's output by its router probability, only for the
 * tokens that were sent to that expert.
 */
export function gateRows(a: Mat, w: Mat, col: number, mask: number[]): Mat {
  const { r, c } = a;
  const o = node(r, c, [a, w]);
  for (let i = 0; i < r; i++) {
    const s = mask[i] ? w.d[i * w.c + col] : 0;
    for (let j = 0; j < c; j++) o.d[i * c + j] = a.d[i * c + j] * s;
  }
  o.back = () => {
    for (let i = 0; i < r; i++) {
      if (!mask[i]) continue;
      const s = w.d[i * w.c + col];
      let gs = 0;
      for (let j = 0; j < c; j++) {
        a.g[i * c + j] += o.g[i * c + j] * s;
        gs += o.g[i * c + j] * a.d[i * c + j];
      }
      w.g[i * w.c + col] += gs;
    }
  };
  return o;
}
