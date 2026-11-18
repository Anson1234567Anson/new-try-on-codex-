import { describe, expect, it } from 'vitest';
import { GPT, sgdStep, DEFAULT_CONFIG } from '../src/model/gpt';
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
    ];
    for (const v of variants) {
      const model = new GPT({ ...DEFAULT_CONFIG, ...v }, rng(7));
      const batch = ['emma', 'noah', 'olivia'].map(example);
      model.loss(batch, true);
      const rand = rng(3);
      for (const [name, m] of Object.entries(model.p)) {
        for (let n = 0; n < 6; n++) {
          const i = Math.floor(rand() * m.d.length);
          const analytic = m.g[i];
          const keep = m.d[i], eps = 1e-5;
          m.d[i] = keep + eps; const up = model.loss(batch);
          m.d[i] = keep - eps; const down = model.loss(batch);
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
