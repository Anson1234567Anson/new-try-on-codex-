// Trains the small model that the lessons inspect, and saves its weights.
// Run with: npm run train
import { writeFileSync } from 'node:fs';
import { GPT, Adam, DEFAULT_CONFIG } from '../src/model/gpt';
import { TRAIN_NAMES, VAL_NAMES, example, rng } from '../src/model/data';

// trained only on the training split, so lessons can test it on names it has never seen
const all = TRAIN_NAMES.map(example);
const val = VAL_NAMES.map(example);
const model = new GPT(DEFAULT_CONFIG, rng(1));
const opt = new Adam(model, 0.01);
const rand = rng(2);
const STEPS = 6000;
for (let i = 1; i <= STEPS; i++) {
  opt.lr = 0.01 * (1 - (0.9 * i) / STEPS);
  model.loss(Array.from({ length: 32 }, () => all[Math.floor(rand() * all.length)]), true);
  opt.step();
  if (i % 1000 === 0) console.log(`step ${i}  val loss ${model.loss(val).toFixed(3)}`);
}
const s = rng(5);
console.log('samples:', Array.from({ length: 10 }, () => model.sample(s)).join(', '));
writeFileSync(new URL('../src/model/weights.json', import.meta.url), JSON.stringify(model.toJSON()));
console.log(`saved ${model.paramCount()} weights`);
