import { describe, expect, it } from 'vitest';
import { GPT, sgdStep, DEFAULT_CONFIG } from '../src/model/gpt';
import { crossEntropy } from '../src/model/tensor';
import { NAMES, CHARS, CTX, example, rng, encode, decode } from '../src/model/data';

describe('data', () => {
  it('names use only the vocabulary and fit the context', () => {
    for (const n of NAMES) {
      expect(n.length).toBeLessThanOrEqual(CTX - 1);
      for (const ch of n) expect(CHARS.indexOf(ch)).toBeGreaterThan(0);
    }
  });
  it('encodes and decodes', () => {
    expect(encode('.emma.')).toEqual([0, 5, 13, 13, 1, 0]);
    expect(decode([0, 5, 13, 13, 1, 0])).toBe('.emma.');
    expect(example('emma')).toEqual({ x: [0, 5, 13, 13, 1], y: [5, 13, 13, 1, 0] });
  });
});

describe('model', () => {
  it('backprop matches a numerical gradient for every kind of weight', () => {
    const variants = [
      { heads: 1 }, { heads: 2 }, { layers: 2, norm: true }, { noPos: true },
      { experts: 4, topK: 1 }, { experts: 4, topK: 2, layers: 2 },
    ];
    for (const v of variants) {
      const model = new GPT({ ...DEFAULT_CONFIG, ...v }, rng(7));
      const batch = ['emma', 'noah', 'olivia'].map(example);
      model.loss(batch, true);
      // the trained objective: prediction loss plus any load-balancing loss
      const objective = () => batch.reduce((sum, ex) => {
        const tr = model.forward(ex.x);
        let v = crossEntropy(tr.logits, ex.y).d[0];
        for (const b of tr.blocks) if (b.balanceLoss) v += 0.01 * b.balanceLoss.d[0];
        return sum + v / batch.length;
      }, 0);
      const rand = rng(3);
      for (const [name, m] of Object.entries(model.p)) {
        for (let n = 0; n < 6; n++) {
          const i = Math.floor(rand() * m.d.length);
          const analytic = m.g[i];
          const keep = m.d[i], eps = 1e-5;
          m.d[i] = keep + eps; const up = objective();
          m.d[i] = keep - eps; const down = objective();
          m.d[i] = keep;
          const numeric = (up - down) / (2 * eps);
          expect(Math.abs(analytic - numeric), `${name}[${i}] ${JSON.stringify(v)}`).toBeLessThan(1e-6);
        }
      }
    }
  });

  it('LoRA gradients match a numerical gradient, and only adapters train', () => {
    const model = new GPT(DEFAULT_CONFIG, rng(7));
    model.addLoRA(['q0', 'u'], 2, rng(9));
    for (const ad of Object.values(model.lora)) for (let i = 0; i < ad.B.d.length; i++) ad.B.d[i] = 0.1 * Math.sin(i);
    const batch = ['emma', 'noah'].map(example);
    model.loss(batch, true);
    expect(model.params().length).toBe(4);
    for (const ad of Object.values(model.lora))
      for (const m of [ad.A, ad.B])
        for (let i = 0; i < m.d.length; i += 3) {
          const keep = m.d[i], eps = 1e-5;
          m.d[i] = keep + eps; const up = model.loss(batch);
          m.d[i] = keep - eps; const down = model.loss(batch);
          m.d[i] = keep;
          expect(Math.abs(m.g[i] - (up - down) / (2 * eps))).toBeLessThan(1e-6);
        }
  });

  it('mixture of experts: each token uses k experts and only their outputs count', () => {
    const model = new GPT({ ...DEFAULT_CONFIG, experts: 4, topK: 1 }, rng(4));
    const tr = model.forward(example('olivia').x);
    const b = tr.blocks[0];
    expect(b.chosen!.every((c) => c.length === 1)).toBe(true);
    for (let i = 0; i < b.router!.r; i++) expect(b.router!.row(i).reduce((a, c) => a + c, 0)).toBeCloseTo(1, 10);
    // the block output equals the chosen expert's output × its router probability
    const i = 2, e = b.chosen![i][0];
    const hid = b.expertHidden![e].row(i);
    const w2 = model.p['w2_e' + e], b2 = model.p['b2_e' + e];
    for (let j = 0; j < 16; j++) {
      let y = b2.d[j];
      for (let h = 0; h < hid.length; h++) y += hid[h] * w2.d[h * 16 + j];
      expect(b.mlpOut.row(i)[j]).toBeCloseTo(y * b.router!.row(i)[e], 10);
    }
  });

  it('an untrained model is about as wrong as a uniform guess', () => {
    const model = new GPT(DEFAULT_CONFIG, rng(1));
    const l = model.loss(NAMES.map(example));
    expect(l).toBeGreaterThan(Math.log(27) - 0.3);
    expect(l).toBeLessThan(Math.log(27) + 0.6);
  });

  it('attention rows sum to 1 and never look ahead', () => {
    const model = new GPT(DEFAULT_CONFIG, rng(1));
    const a = model.forward(example('emma').x).att[0];
    for (let i = 0; i < a.r; i++) {
      const row = a.row(i);
      expect(row.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 10);
      for (let j = i + 1; j < a.c; j++) expect(row[j]).toBe(0);
    }
  });

  it('training lowers the loss', () => {
    const model = new GPT(DEFAULT_CONFIG, rng(1));
    const all = NAMES.map(example);
    const before = model.loss(all);
    const rand = rng(2);
    for (let i = 0; i < 200; i++) {
      model.loss(Array.from({ length: 16 }, () => all[Math.floor(rand() * all.length)]), true);
      sgdStep(model, 0.5);
    }
    expect(model.loss(all)).toBeLessThan(before - 0.5);
  });
});
