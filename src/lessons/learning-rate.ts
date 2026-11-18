import type { Lesson } from './types';
import type { Item, Tone } from '../engine/scene';
import { t } from '../i18n';
import exp from '../model/experiments.json';

// Three real 2,000-step Adam runs from experiments.json: a low fixed learning rate, a high fixed one,
// and warmup + cosine decay with the same peak as the high one.

const S = exp.schedules;
const EVERY = S.low.every;
const TOTAL = S.steps;
type Run = { val: (number | null)[]; train: (number | null)[]; lr: number[] };
const RUNS = { low: S.low as Run, high: S.high as Run, wc: S.warmupCosine as Run };
type Name = keyof typeof RUNS;
const TONE: Record<Name, Tone> = { low: 'b', high: 'bad', wc: 'a' };

// Where the high run breaks: the first log interval whose training loss is missing (NaN) or jumps above 3.
const BREAK = RUNS.high.train.findIndex((v, i) => i > 0 && (v === null || v > 3));
const BREAK_LO = (BREAK - 1) * EVERY, BREAK_HI = BREAK * EVERY;
const WC_LR_AT_BREAK = RUNS.wc.lr[BREAK];
const PEAK = Math.max(...RUNS.wc.lr);
const WARM = RUNS.wc.lr.findIndex((v) => v >= PEAK * 0.999) * EVERY; // first logged step at the peak
const LAST = RUNS.wc.lr.length - 1;
const RATIO = Math.round(RUNS.high.lr[0] / RUNS.low.lr[0]);
const last = (a: (number | null)[]) => a[a.length - 1] as number;
const at = (a: (number | null)[], step: number) => a[step / EVERY];
const f3 = (v: number | null) => (v === null || !Number.isFinite(v) ? 'NaN' : v > 99 ? Math.round(v).toLocaleString('en-US') : v.toFixed(3));
const fl = (v: number) => +v.toPrecision(2);

// what each step shows: which runs, and how much of the run
const SHOW: { runs: Name[]; xmax: number }[] = [
  { runs: ['low', 'high'], xmax: 500 },
  { runs: ['low', 'high'], xmax: TOTAL },
  { runs: ['low', 'high'], xmax: TOTAL },
  { runs: ['wc'], xmax: TOTAL },
  { runs: ['wc'], xmax: TOTAL },
  { runs: ['low', 'high', 'wc'], xmax: TOTAL },
  { runs: ['high', 'wc'], xmax: TOTAL },
  { runs: ['low', 'high', 'wc'], xmax: TOTAL },
];

export const learningRate: Lesson = {
  id: 'learning-rate',
  title: { en: 'Learning Rate and Warmup', zh: '學習率與暖身' },
  blurb: {
    en: 'Three real training runs: too careful, too bold, and a schedule that starts small, peaks, then slows down.',
    zh: '三次真實訓練：太保守、太大膽，以及一個先小、再衝高、最後放慢的排程。',
  },
  minutes: 11,
  level: 'beginner',
  learn: [
    { text: { en: 'The trade-off between a low and a high learning rate', zh: '學習率低與高之間的取捨' }, step: 0 },
    { text: { en: 'How a too-high learning rate fails, late and suddenly', zh: '學習率太高時，如何突然在後期崩潰' }, step: 1 },
    { text: { en: 'Warmup: start with small steps', zh: '暖身：從小步伐開始' }, step: 3 },
    { text: { en: 'Decay: smaller steps to settle at the end', zh: '衰減：最後用更小的步伐穩定下來' }, step: 4 },
    { text: { en: 'What a schedule changed in our real run, honestly', zh: '排程在我們的真實訓練中到底改變了什麼' }, step: 6 },
  ],
  steps: [
    {
      short: { en: 'The choice', zh: '抉擇' }, chapter: 'problem',
      title: { en: 'Careful or bold?', zh: '保守還是大膽？' },
      body: [
        { en: `The <b>learning rate</b> is how big a step each weight takes. Here are two real Adam runs on our model: a careful one at ${RUNS.low.lr[0]} and a bold one at ${RUNS.high.lr[0]}, about ${RATIO}× larger. The chart shows the loss on held-out names.`, zh: `<b>學習率</b>決定每個權重一步跨多大。這是我們模型的兩次真實 Adam 訓練：保守的用 ${RUNS.low.lr[0]}，大膽的用 ${RUNS.high.lr[0]}，大約大了 ${RATIO} 倍。圖中是模型在沒練過的名字上的損失。` },
        { en: `The bold run races ahead: after 25 steps its loss is ${f3(at(RUNS.high.val, 25))}, against ${f3(at(RUNS.low.val, 25))} for the careful one. The lower chart shows each run’s learning rate.`, zh: `大膽的那次一路領先：25 步後損失是 ${f3(at(RUNS.high.val, 25))}，保守的那次還在 ${f3(at(RUNS.low.val, 25))}。下方的圖是每次訓練所用的學習率。` },
      ],
      changed: [{ sign: '+', text: { en: 'Two runs, first 500 steps', zh: '兩次訓練，前 500 步' } }],
    },
    {
      short: { en: 'Blow-up', zh: '崩潰' }, chapter: 'problem',
      title: { en: 'Case: the bold run suddenly breaks', zh: '案例：大膽的那次突然崩潰' },
      body: [
        { en: `Let both runs continue to ${TOTAL.toLocaleString()} steps. For a long time the bold run looks fine. Then, between step ${BREAK_LO.toLocaleString()} and ${BREAK_HI.toLocaleString()}, its loss shoots up (the red line hits the top of the chart; the held-out loss reached ${f3(RUNS.high.val[BREAK])}).`, zh: `讓兩次訓練都跑到 ${TOTAL.toLocaleString()} 步。大膽的那次很長一段時間看起來都沒問題，然後在第 ${BREAK_LO.toLocaleString()} 到 ${BREAK_HI.toLocaleString()} 步之間，損失突然暴衝（紅線衝到圖表頂端；沒練過的名字上的損失到了 ${f3(RUNS.high.val[BREAK])}）。` },
        { en: `One step jumped too far, landed somewhere bad, and the following steps made it worse. It recovers only partly, ending at ${f3(last(RUNS.high.val))}.`, zh: `某一步跨得太遠，落到很糟的地方，接下來的步伐又讓情況更糟。之後只恢復了一部分，最後停在 ${f3(last(RUNS.high.val))}。` },
      ],
      callout: { kind: 'note', text: { en: 'A run can look healthy for a long time before it breaks. Large training runs are watched constantly for exactly this.', zh: '一次訓練可能看起來健康了很久，然後才崩潰。大型訓練之所以要隨時盯著，正是為了這種狀況。' } },
      changed: [{ sign: '~', text: { en: `Bold run breaks near step ${BREAK_HI.toLocaleString()}`, zh: `大膽的那次在第 ${BREAK_HI.toLocaleString()} 步附近崩潰` } }],
    },
    {
      short: { en: 'Careful', zh: '保守' }, chapter: 'problem',
      title: { en: 'The careful run: slow, but safe', zh: '保守的那次：慢，但安全' },
      body: [
        { en: `The careful run never breaks. It is slow at the start, but keeps creeping down and ends at ${f3(last(RUNS.low.val))}.`, zh: `保守的那次從沒出事。它一開始很慢，但一路慢慢往下，最後停在 ${f3(last(RUNS.low.val))}。` },
        { en: 'So we want both: big steps for speed, and small steps for safety. The trick is that we do not need the same learning rate for the whole run.', zh: '所以我們兩者都想要：大步伐求快，小步伐求穩。關鍵是：整次訓練不必一直用同一個學習率。' },
      ],
      changed: [{ sign: '~', text: { en: `Same runs, full ${TOTAL.toLocaleString()} steps`, zh: `同樣兩次訓練，完整 ${TOTAL.toLocaleString()} 步` } }],
    },
    {
      short: { en: 'Warmup', zh: '暖身' }, chapter: 'technique',
      title: { en: 'Warmup: start small', zh: '暖身：從小開始' },
      body: [
        { en: `A <b>schedule</b> changes the learning rate during training. It starts with <b>warmup</b>: the rate climbs in a straight line from almost 0 to its peak of ${PEAK} over the first ${WARM} steps.`, zh: `<b>排程（schedule）</b>會在訓練過程中改變學習率。它從<b>暖身（warmup）</b>開始：在前 ${WARM} 步，學習率從接近 0 直線爬升到最高點 ${PEAK}。` },
        { en: 'Why? At the very start the weights are random, so the gradients point in rough, unreliable directions, and Adam’s running averages have almost no history yet. Small steps keep one bad early gradient from doing real damage.', zh: '為什麼？訓練剛開始時權重是隨機的，梯度指的方向粗糙又不可靠，Adam 的移動平均也幾乎還沒有累積任何紀錄。用小步伐，就不會讓一個早期的壞梯度造成真正的破壞。' },
      ],
      changed: [{ sign: '+', text: { en: `Warmup: 0 → ${PEAK} over ${WARM} steps`, zh: `暖身：${WARM} 步內 0 → ${PEAK}` } }],
    },
    {
      short: { en: 'Decay', zh: '衰減' }, chapter: 'technique',
      title: { en: 'Decay: slow down to settle', zh: '衰減：放慢腳步，穩定下來' },
      body: [
        { en: `After the peak, the rate slides down along a smooth curve (a <b>cosine</b> shape) to ${fl(RUNS.wc.lr[LAST])} at the end, a tenth of the peak.`, zh: `過了最高點之後，學習率沿著一條平滑曲線（<b>餘弦（cosine）</b>形狀）往下滑，最後降到 ${fl(RUNS.wc.lr[LAST])}，是最高點的十分之一。` },
        { en: 'Early on, big steps cover ground quickly. Near the end the model is close to a good spot, and big steps would just bounce around it. Smaller steps let it settle in.', zh: '前期用大步伐快速前進；接近尾聲時，模型已經離好的位置不遠，大步伐只會在附近彈來彈去。步伐變小，才能穩穩落定。' },
      ],
      changed: [{ sign: '+', text: { en: `Cosine decay to ${fl(RUNS.wc.lr[LAST])}`, zh: `餘弦衰減到 ${fl(RUNS.wc.lr[LAST])}` } }],
    },
    {
      short: { en: 'Schedule', zh: '排程' }, chapter: 'solution',
      title: { en: 'Warmup + cosine: stable and the best of the three', zh: '暖身 + 餘弦：穩定，也是三者中最好的' },
      body: [
        { en: `The teal run uses the schedule, with the same peak (${PEAK}) as the run that broke. It never breaks, and ends with the lowest training loss (${f3(last(RUNS.wc.train))}, against ${f3(last(RUNS.low.train))} for the careful run).`, zh: `青色那次用了排程，最高學習率（${PEAK}）和崩潰的那次一樣。它從沒出事，最後的訓練損失也最低（${f3(last(RUNS.wc.train))}，保守的那次是 ${f3(last(RUNS.low.train))}）。` },
        { en: `Be honest about the size of the win: on held-out names it ends at ${f3(last(RUNS.wc.val))}, the careful run at ${f3(last(RUNS.low.val))}. On a model this small the two end up practically tied.`, zh: `老實說，贏得不多：在沒練過的名字上，它最後是 ${f3(last(RUNS.wc.val))}，保守的那次是 ${f3(last(RUNS.low.val))}。在這麼小的模型上，兩者幾乎打平。` },
      ],
      changed: [{ sign: '+', text: { en: 'Third run: warmup + cosine', zh: '第三次訓練：暖身 + 餘弦' } }],
    },
    {
      short: { en: 'What saved it', zh: '誰救了它' }, chapter: 'problem',
      title: { en: 'Case: what actually saved the schedule here?', zh: '案例：這裡到底是什麼救了排程？' },
      body: [
        { en: `The bold run broke late, near step ${BREAK_HI.toLocaleString()}, not during the first steps. By then the schedule had already decayed to ${fl(WC_LR_AT_BREAK)}, well below ${RUNS.high.lr[0]}. On our tiny task it was the <i>decay</i> that kept it safe.`, zh: `大膽的那次是在後期、第 ${BREAK_HI.toLocaleString()} 步附近崩潰的，而不是一開始。那時排程的學習率已經降到 ${fl(WC_LR_AT_BREAK)}，遠低於 ${RUNS.high.lr[0]}。在我們的小任務上，真正保住它的是<i>衰減</i>。` },
        { en: 'Warmup made no visible difference here: our small model did not break early even without it. Larger models are much more fragile at the start, which is why warmup is standard practice for them.', zh: '暖身在這裡看不出明顯效果：我們的小模型就算沒有暖身，一開始也沒出事。大型模型在訓練初期脆弱得多，所以對它們來說，暖身是標準做法。' },
      ],
      callout: { kind: 'note', text: { en: 'The original 2017 Transformer paper already used a warmup of 4,000 steps.', zh: '2017 年最早提出 Transformer 的論文，就已經使用了 4,000 步的暖身。' } },
      changed: [{ sign: '~', text: { en: `Learning rate at the break: ${RUNS.high.lr[0]} vs ${fl(WC_LR_AT_BREAK)}`, zh: `崩潰那一刻的學習率：${RUNS.high.lr[0]} 對 ${fl(WC_LR_AT_BREAK)}` } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Change the step size as training goes', zh: '隨著訓練調整步伐大小' },
      body: [
        { en: '<b>Too high</b>: fast at first, can break suddenly, even late.<br><b>Too low</b>: safe but slow.<br><b>Warmup</b>: small steps while the gradients are still unreliable.<br><b>Decay</b>: shrinking steps so the model settles.<br>On our tiny model the schedule helps a little; on large models it matters a lot more.', zh: '<b>太高</b>：一開始快，但可能突然崩潰，甚至在後期。<br><b>太低</b>：安全但慢。<br><b>暖身</b>：梯度還不可靠時，先走小步。<br><b>衰減</b>：步伐越來越小，讓模型穩定下來。<br>在我們的小模型上排程只幫了一點忙；在大型模型上就重要得多。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Overfitting and Validation. A low training loss is not the goal. How do we know the model works on names it has never seen?', zh: '下一課：過度擬合與驗證。訓練損失低並不是目標。我們怎麼知道模型在沒看過的名字上也表現良好？' } },
    },
  ],

  scene(step) {
    const items: Item[] = [];
    const { runs, xmax } = SHOW[step];
    const n = xmax / EVERY + 1;
    const cut = (a: (number | null)[]) => a.slice(0, n).map((v) => (v === null ? NaN : v));
    const tight = xmax === 500;
    items.push({
      k: 'loss', t: 'chart', x: 90, y: 40, w: 760, h: 230, ymin: tight ? 2.2 : 2.0, ymax: tight ? 3.0 : 4.0, xmax, every: EVERY,
      series: runs.map((r) => ({ pts: cut(RUNS[r].val), tone: TONE[r] })),
      xlabel: t({ en: 'training steps', zh: '訓練步數' }), ylabel: t({ en: 'loss on held-out names (lower is better)', zh: '沒練過的名字上的損失（越低越好）' }),
    });
    items.push({
      k: 'lr', t: 'chart', x: 90, y: 345, w: 760, h: 120, ymin: 0, ymax: 6, xmax, every: EVERY,
      series: runs.map((r) => ({ pts: RUNS[r].lr.slice(0, n).map((v) => v * 100), tone: TONE[r] })),
      xlabel: '', ylabel: t({ en: 'learning rate (× 0.01)', zh: '學習率（× 0.01）' }),
    });
    if (step === 1 || step === 6) {
      const x = 90 + (BREAK_HI / TOTAL) * 760;
      items.push({ k: 'brk', t: 'arrow', x1: x, y1: 300, x2: x, y2: 274, tone: 'bad' });
      items.push({ k: 'brkT', t: 'text', x, y: 318, anchor: 'middle', size: 13.5, tone: 'bad', text: t({ en: `breaks: step ${BREAK_LO.toLocaleString()}–${BREAK_HI.toLocaleString()}`, zh: `崩潰：第 ${BREAK_LO.toLocaleString()}–${BREAK_HI.toLocaleString()} 步` }) });
    }
    if (step === 6) {
      const x = 90 + (BREAK_HI / TOTAL) * 760;
      const y = 345 + 120 - (WC_LR_AT_BREAK * 100 / 6) * 120;
      items.push({ k: 'wcl', t: 'text', x: x + 10, y: y - 8, size: 13.5, tone: 'a', text: t({ en: `schedule here: ${fl(WC_LR_AT_BREAK)}`, zh: `此時排程：${fl(WC_LR_AT_BREAK)}` }) });
    }
    if (step === 3 || step === 4) {
      const x = 90 + (WARM / TOTAL) * 760;
      items.push({ k: 'wm', t: 'text', x: step === 3 ? x + 8 : 860, y: 360, anchor: step === 3 ? 'start' : 'end', size: 13.5, tone: 'a',
        text: step === 3 ? t({ en: `← warmup: ${WARM} steps`, zh: `← 暖身：${WARM} 步` }) : t({ en: `cosine decay to ${fl(RUNS.wc.lr[LAST])} →`, zh: `餘弦衰減到 ${fl(RUNS.wc.lr[LAST])} →` }) });
    }

    const metric = (r: Name): { label: { en: string; zh: string }; value: string } => {
      const names = { low: { en: 'Careful', zh: '保守' }, high: { en: 'Bold', zh: '大膽' }, wc: { en: 'Schedule', zh: '排程' } }[r];
      const v = step === 0 ? at(RUNS[r].val, 100) : last(RUNS[r].val);
      return { label: { en: `${names.en} · ${step === 0 ? 'step 100' : 'final'}`, zh: `${names.zh} · ${step === 0 ? '第 100 步' : '最後'}` }, value: f3(v) };
    };
    const metrics = [
      ...(['low', 'high', 'wc'] as Name[]).filter((r) => runs.includes(r)).map(metric),
      { label: { en: 'Peak learning rate', zh: '最高學習率' }, value: String(runs.includes('high') ? RUNS.high.lr[0] : PEAK) },
    ];

    const names: Record<Name, { en: string; zh: string }> = {
      low: { en: `careful · ${RUNS.low.lr[0]}`, zh: `保守 · ${RUNS.low.lr[0]}` },
      high: { en: `bold · ${RUNS.high.lr[0]}`, zh: `大膽 · ${RUNS.high.lr[0]}` },
      wc: { en: `warmup + cosine · peak ${PEAK}`, zh: `暖身 + 餘弦 · 最高 ${PEAK}` },
    };
    return {
      metrics,
      items,
      legend: runs.map((r) => ({ tone: TONE[r] === 'bad' ? 'bad' : TONE[r] === 'a' ? 'a' : 'b', text: names[r] })),
    };
  },
};
