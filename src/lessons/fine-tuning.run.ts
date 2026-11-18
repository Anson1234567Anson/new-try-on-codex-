// Helpers shared by the "Pretraining vs Fine-Tuning" and "LoRA" lessons: the same fine-tuning
// runs as scripts/experiments.ts (the course model fine-tuned on 40 names ending in "a"),
// replayed in the browser in small slices, so a lesson can stop at any step or try any rank.
import { GPT, Adam, type Weights } from '../model/gpt';
import { TRAIN_NAMES, VAL_NAMES, example, rng } from '../model/data';
import weights from '../model/weights.json';

type Ex = { x: number[]; y: number[] };
/** The weights the experiment's LoRA run adapted. */
export const ADAPTED = ['q0', 'k0', 'v0', 'o', 'w1', 'w2', 'u'];
export const BATCH = 32;
const CHUNK = 8; // names per slice of a training step (a batch of 32 is done in 4 slices)
const VCHUNK = 31; // held-out names per slice of an evaluation

let sets: { base: GPT; train: Ex[]; val: Ex[]; trainNames: string[] } | undefined;
export const FT_SETS = () => sets ??= (() => {
  const trainNames = TRAIN_NAMES.filter((n) => n.endsWith('a')).slice(0, 40); // the same 40 the experiment used
  return { base: GPT.fromJSON(weights as Weights), train: trainNames.map(example), val: VAL_NAMES.filter((n) => n.endsWith('a')).map(example), trainNames };
})();

export interface FtRun {
  model: GPT; opt: Adam; rand: () => number;
  step: number; // finished training steps
  val: number[]; // held-out loss after 0, every, 2·every, … steps
  batch: Ex[] | null; pos: number; acc: Float64Array[] | null; // the step in progress
  ev: { pos: number; sum: number } | null; // the evaluation in progress
}

/** rank 0 = change every weight (full fine-tuning); rank r = LoRA adapters of rank r. */
export function newRun(rank: number, lr: number): FtRun {
  const model = FT_SETS().base.clone();
  if (rank > 0) model.addLoRA(ADAPTED, rank, rng(3));
  // the experiment draws its batches from rng(2), like every run in scripts/experiments.ts
  return { model, opt: new Adam(model, lr), rand: rng(2), step: 0, val: [], batch: null, pos: 0, acc: null, ev: { pos: 0, sum: 0 } };
}

/** Do one small slice of work. Returns false once `steps` steps are trained and logged. */
export function slice(r: FtRun, steps: number, every: number): boolean {
  const { train, val } = FT_SETS();
  if (r.ev) {
    const part = val.slice(r.ev.pos, r.ev.pos + VCHUNK);
    r.ev.sum += r.model.loss(part) * part.length;
    r.ev.pos += part.length;
    if (r.ev.pos >= val.length) { r.val.push(r.ev.sum / val.length); r.ev = null; }
    return true;
  }
  if (r.step >= steps) return false;
  if (!r.batch) {
    r.batch = Array.from({ length: BATCH }, () => train[Math.floor(r.rand() * train.length)]);
    r.pos = 0;
    r.acc = r.model.params().map((m) => new Float64Array(m.g.length));
  }
  // gradient of the batch average = the average of the 4 slices' gradients
  r.model.loss(r.batch.slice(r.pos, r.pos + CHUNK), true);
  r.model.params().forEach((m, i) => { const a = r.acc![i]; for (let j = 0; j < a.length; j++) a[j] += m.g[j] * (CHUNK / BATCH); });
  r.pos += CHUNK;
  if (r.pos >= BATCH) {
    r.model.params().forEach((m, i) => m.g.set(r.acc![i]));
    r.opt.step();
    r.step++;
    r.batch = null; r.acc = null;
    if (r.step % every === 0) r.ev = { pos: 0, sum: 0 };
  }
  return true;
}

/** Run to the end at once (for small runs a lesson needs right away). */
export function finish(r: FtRun, steps: number, every: number): FtRun { while (slice(r, steps, every)); return r; }

/** Share of 400 written names that end in "a", with the experiment's seed. */
export function endsA(m: GPT): number {
  const r = rng(11);
  const s = Array.from({ length: 400 }, () => m.sample(r)).filter(Boolean);
  return s.filter((n) => n.endsWith('a')).length / s.length;
}
/** 8 names, with the same seed the experiment used for its samples. */
export function writeNames(m: GPT): string[] { const r = rng(5); return Array.from({ length: 8 }, () => m.sample(r) || '·'); }
