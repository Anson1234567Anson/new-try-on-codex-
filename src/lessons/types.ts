import type { L } from '../i18n';
import type { Item } from '../engine/scene';

export type Chapter = 'problem' | 'technique' | 'solution' | 'recap';

export interface Step {
  short: L; // label on the timeline
  chapter: Chapter;
  title: L;
  body: L[]; // short paragraphs; may contain <code> and <b>
  callout?: { kind: 'insight' | 'note' | 'next'; text: L };
  changed?: { sign: '+' | '~'; text: L }[];
}

export interface Scene {
  metrics: { label: L; value: string }[];
  items: Item[];
  legend?: { tone: 'pos' | 'neg' | 'heat' | 'a' | 'b' | 'bad'; text: L }[];
}

export interface Lesson {
  id: string;
  title: L;
  blurb: L;
  minutes: number;
  level: 'beginner' | 'intermediate';
  learn: { text: L; step: number }[]; // ticked once the learner reaches `step` (0-based)
  steps: Step[];
  scene(step: number): Scene;
  /** For lessons that compute over time: advance a little; return true while there is more to do. */
  tick?(step: number): boolean;
  /** Start the current step's computation again from scratch. */
  reset?(): void;
}
