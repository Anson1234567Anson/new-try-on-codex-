import { describe, expect, it } from 'vitest';
import { LESSONS, MODULES } from '../src/course';
import { setLang, type L } from '../src/i18n';

const outline = MODULES.flatMap((m) => m.lessons.map((l) => l.id));
const filled = (s: L | undefined, where: string, chinese = false) => {
  if (!s) return;
  expect(s.en?.trim(), `${where} (en)`).toBeTruthy();
  expect(s.zh?.trim(), `${where} (zh)`).toBeTruthy();
  if (chinese) expect(/[一-鿿]/.test(s.zh), `${where} (zh) has Chinese text`).toBe(true);
};

describe('lessons', () => {
  it('every written lesson is in the outline', () => {
    for (const id of Object.keys(LESSONS)) expect(outline, id).toContain(id);
  });

  for (const lesson of Object.values(LESSONS)) {
    describe(lesson.id, () => {
      it('has text in both languages', () => {
        filled(lesson.title, 'title', true); filled(lesson.blurb, 'blurb', true);
        lesson.learn.forEach((x, i) => {
          filled(x.text, `learn ${i}`);
          expect(x.step, `learn ${i} step`).toBeLessThan(lesson.steps.length);
        });
        lesson.steps.forEach((s, i) => {
          const w = `step ${i + 1}`;
          filled(s.short, `${w} short`); filled(s.title, `${w} title`, true);
          s.body.forEach((b, j) => filled(b, `${w} body ${j}`, true));
          filled(s.callout?.text, `${w} callout`);
          s.changed?.forEach((c, j) => filled(c.text, `${w} changed ${j}`));
        });
        expect(lesson.steps[lesson.steps.length - 1].chapter).toBe('recap');
      });

      for (const lang of ['en', 'zh'] as const) {
        it(`draws every step (${lang})`, () => {
          setLang(lang);
          lesson.reset?.();
          lesson.steps.forEach((_, i) => {
            const sc = lesson.scene(i);
            const keys = sc.items.map((it) => it.k);
            expect(new Set(keys).size, `step ${i + 1}: item keys must be unique`).toBe(keys.length);
            const text = JSON.stringify(sc);
            expect(text, `step ${i + 1}`).not.toMatch(/undefined|\[object Object\]/);
            for (const m of sc.metrics) filled(m.label, `step ${i + 1} metric`);
            for (const it of sc.items)
              if ('x' in it && 'y' in it) {
                expect(Number.isFinite(it.x) && Number.isFinite(it.y), `step ${i + 1} ${it.k} position`).toBe(true);
              }
          });
          setLang('en');
        });
      }
    });
  }
});
