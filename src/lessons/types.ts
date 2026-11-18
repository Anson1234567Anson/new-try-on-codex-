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

/**
 * A control the learner can change on this step. Its current value is passed
 * back to `scene(step, input)` as `input[key]`, and the diagram redraws live.
 */
export type Control =
  | { key: string; kind: 'slider'; label: L; min: number; max: number; step: number; value: number; format?: (v: number) => string }
  | { key: string; kind: 'text'; label: L; value: string; maxLength?: number; pattern?: string; placeholder?: L }
  | { key: string; kind: 'toggle'; label: L; value: boolean }
  | { key: string; kind: 'choice'; label: L; options: { value: string; label: L }[]; value: string };

export type Input = Record<string, number | string | boolean>;

export interface Scene {
  metrics: { label: L; value: string }[];
  items: Item[];
  legend?: { tone: 'pos' | 'neg' | 'heat' | 'a' | 'b' | 'bad'; text: L }[];
  /** Optional controls for this step. `value` is the starting value; the learner's changes are kept while they stay on the lesson. */
  controls?: Control[];
}

export interface Lesson {
  id: string;
  title: L;
  blurb: L;
  minutes: number;
  level: 'beginner' | 'intermediate';
  learn: { text: L; step: number }[]; // ticked once the learner reaches `step` (0-based)
  steps: Step[];
  /** `input` holds the learner's current control values (empty until they touch a control). */
  scene(step: number, input?: Input): Scene;
  /**
   * For lessons that compute over time: advance a little; return true while there is more to do.
   * `input` is the step's current control values; when the learner changes one, tick is called again.
   */
  tick?(step: number, input: Input): boolean;
  /** Start the current step's computation again from scratch. */
  reset?(): void;
}
