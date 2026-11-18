// The course's running example: a tiny model that learns to write first names.

// About 3,000 English first names (2–7 letters), from the MIT-licensed
// "human-names" package by Alessandro Minoccheri, lower-cased and shuffled.
import names from './names.json';

export const NAMES: string[] = names;

/** Every fifth name is held back to check the model on names it never trained on. */
export const VAL_NAMES = NAMES.filter((_, i) => i % 5 === 0);
export const TRAIN_NAMES = NAMES.filter((_, i) => i % 5 !== 0);

/** Vocabulary: '.' marks the start and the end of a name, then a–z. */
export const CHARS = '.abcdefghijklmnopqrstuvwxyz';
export const VOCAB = CHARS.length;
export const CTX = 8;

export const encode = (s: string): number[] => [...s].map((ch) => CHARS.indexOf(ch));
export const decode = (ids: number[]): string => ids.map((i) => CHARS[i]).join('');

/** One training example: the model reads `x` and should predict `y`, one step ahead. */
export function example(name: string): { x: number[]; y: number[] } {
  const seq = encode('.' + name + '.');
  return { x: seq.slice(0, -1).slice(0, CTX), y: seq.slice(1).slice(0, CTX) };
}

/** Small seeded random number generator, so every run is repeatable. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand: () => number): number {
  return Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
}
