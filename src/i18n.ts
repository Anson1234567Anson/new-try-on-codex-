// Every piece of text in the course exists in English and Traditional Chinese.

export type Lang = 'en' | 'zh';
export type L = { en: string; zh: string };

let current: Lang = 'en';
export const getLang = (): Lang => current;
export const setLang = (l: Lang): void => { current = l; };
export const t = (s: L | string): string => (typeof s === 'string' ? s : s[current]);

export const UI = {
  brand: { en: 'Transformers, step by step', zh: 'Transformer 一步一步學' },
  course: { en: 'Course', zh: '課程' },
  lessons: { en: 'lessons', zh: '堂課' },
  soon: { en: 'soon', zh: '即將推出' },
  progress: { en: 'Progress', zh: '進度' },
  lessonOf: { en: 'Lesson {a} of {b}', zh: '第 {a} 課，共 {b} 課' },
  stepOf: { en: 'Step {a} of {b}', zh: '第 {a} 步，共 {b} 步' },
  min: { en: 'min', zh: '分鐘' },
  steps: { en: 'steps', zh: '步' },
  replay: { en: 'Replay', zh: '重播' },
  pause: { en: 'Pause', zh: '暫停' },
  resume: { en: 'Resume', zh: '繼續' },
  back: { en: 'Back', zh: '上一步' },
  next: { en: 'Next', zh: '下一步' },
  finish: { en: 'Finish lesson', zh: '完成本課' },
  inLesson: { en: 'In this lesson', zh: '本課內容' },
  learn: { en: "You'll learn", zh: '你會學到' },
  changed: { en: 'What changed', zh: '有什麼改變' },
  keys: { en: '← → step · R replay · Space pause', zh: '← → 換步 · R 重播 · 空白鍵 暫停' },
  theme: { en: 'Switch light / dark', zh: '切換淺色／深色' },
  realNumbers: {
    en: 'Diagram numbers come from real models: computed live in this tab, or from real training runs saved with the course. Illustrations are labelled.',
    zh: '圖中的數字都來自真正的模型：不是在這個分頁中即時算出來的，就是課程附帶的真實訓練紀錄。示意圖會另外標明。',
  },
  chapter: {
    problem: { en: 'Problem', zh: '問題' },
    technique: { en: 'Technique', zh: '方法' },
    solution: { en: 'Solution', zh: '解法' },
    recap: { en: 'Recap', zh: '回顧' },
  },
  callout: {
    insight: { en: 'Key idea', zh: '關鍵觀念' },
    note: { en: 'Good to know', zh: '補充說明' },
    next: { en: 'Coming up', zh: '接下來' },
  },
  level: {
    beginner: { en: 'Beginner', zh: '入門' },
    intermediate: { en: 'Intermediate', zh: '中階' },
  },
} as const;

export const fill = (s: L, vars: Record<string, string | number>): string =>
  t(s).replace(/\{(\w+)\}/g, (_, k) => String(vars[k]));
