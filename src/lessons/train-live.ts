import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t } from '../i18n';
import { GPT, DEFAULT_CONFIG, Adam, warmupCosine } from '../model/gpt';
import { TRAIN_NAMES, VAL_NAMES, encode, example, rng } from '../model/data';

// A fresh model trained from scratch in the browser with Adam and warmup + cosine decay.
// Work is done in small slices by `tick`, so the page never freezes.

const TOTAL = 1200; // iterations in the whole run
const PEAK = 0.02; // highest learning rate
const WARM = 100; // warmup iterations
const BATCH = 16;
const EVERY = 25; // log the losses every 25 iterations
const EVAL_N = 150; // fixed held-out names used to measure the held-out loss
const CHUNK = 10; // names evaluated per slice of work
const COPY_N = 300; // names written for the copying check
const WORD = '.sofia'; // the word whose attention grid we watch
const UNIFORM = Math.log(27);

type Ex = { x: number[]; y: number[] };
let sets: { train: Ex[]; trainEval: Ex[]; val: Ex[]; valAll: Ex[]; trainSet: Set<string>; valSet: Set<string> } | undefined;
const S = () => sets ??= {
  train: TRAIN_NAMES.map(example),
  trainEval: TRAIN_NAMES.slice(0, EVAL_N).map(example),
  val: VAL_NAMES.slice(0, EVAL_N).map(example),
  valAll: VAL_NAMES.map(example),
  trainSet: new Set(TRAIN_NAMES), valSet: new Set(VAL_NAMES),
};

interface Cfg { peak: number; warm: number; batch: number }
const BASE: Cfg = { peak: PEAK, warm: WARM, batch: BATCH };
const lrOf = (c: Cfg, i: number) => warmupCosine(i, TOTAL, c.peak, c.warm);
const lrAt = (i: number) => lrOf(BASE, i);
const OPS_PER_TOKEN = 6 * 3115; // the Compute lesson's rule: 6 × weights per token
const WORD_IDS = encode(WORD);
const LETTERS = [...WORD];

interface Job { set: Ex[]; pos: number; sum: number; done: (v: number) => void }
interface Run {
  cfg: Cfg; tokens: number; valAll?: number;
  model: GPT; opt: Adam; rand: () => number; iter: number;
  train: number[]; val: number[]; acc: number; accN: number;
  jobs: Job[]; samples: string[]; samplesAt: number;
  att0: (number | null)[][];
  copy: { train: number; val: number; fresh: number; n: number; examples: string[]; rand: () => number } | null;
}
let run: Run | null = null;

const attGrid = (m: GPT) => {
  const a = m.forward(WORD_IDS).att[0];
  return WORD_IDS.map((_, i) => a.row(i).map((v, j) => (j > i ? null : v)));
};
// how focused attention is: the average of each row's largest weight (rows with 3+ letters to choose from)
const focus = (g: (number | null)[][]) => {
  const rows = g.slice(2).map((r) => Math.max(...r.filter((v): v is number => v !== null)));
  return rows.reduce((a, b) => a + b, 0) / rows.length;
};

function evalJob(set: Ex[], done: (v: number) => void): Job { return { set, pos: 0, sum: 0, done }; }

function make(cfg: Cfg): Run {
  const model = new GPT(DEFAULT_CONFIG, rng(1));
  const r: Run = {
    cfg, tokens: 0, model, opt: new Adam(model, 0), rand: rng(2), iter: 0, train: [], val: [], acc: 0, accN: 0,
    jobs: [], samples: [], samplesAt: -1, att0: attGrid(model), copy: null,
  };
  r.jobs.push(evalJob(S().trainEval, (v) => r.train.push(v)), evalJob(S().val, (v) => r.val.push(v)));
  return r;
}
function fresh(): Run { run = make(BASE); return run; }
const get = () => run ?? fresh();

// the learner's own run (the control-panel step), and the previous one as a gray ghost
let own: Run | null = null, ghost: { val: number[]; cfg: Cfg } | null = null;
const cfgOf = (input: Record<string, unknown>): Cfg => ({
  peak: +(10 ** Number(input.lr ?? Math.log10(PEAK))).toPrecision(2),
  warm: Number(input.warm ?? WARM) || 0,
  batch: Number(input.batch ?? BATCH) || BATCH,
});
const same = (a: Cfg, b: Cfg) => a.peak === b.peak && a.warm === b.warm && a.batch === b.batch;
function ownFor(input: Record<string, unknown>): Run {
  const cfg = cfgOf(input);
  if (!own || !same(own.cfg, cfg)) {
    if (own && own.iter > 0) ghost = { val: own.val, cfg: own.cfg };
    own = make(cfg);
  }
  return own;
}

function trainOne(r: Run): void {
  const { train } = S();
  const batch = Array.from({ length: r.cfg.batch }, () => train[Math.floor(r.rand() * train.length)]);
  for (const b of batch) r.tokens += b.y.length;
  r.opt.lr = lrOf(r.cfg, r.iter);
  r.acc += r.model.loss(batch, true); r.accN++;
  r.opt.step();
  r.iter++;
  if (r.iter % EVERY === 0) {
    r.train.push(r.acc / r.accN); r.acc = 0; r.accN = 0;
    r.jobs.push(evalJob(S().val, (v) => r.val.push(v)));
    // at the very end, also measure all 615 held-out names once, comparable with other lessons
    if (r.iter === TOTAL) r.jobs.push(evalJob(S().valAll, (v) => { r.valAll = v; }));
  }
}

function evalChunk(r: Run): void {
  const j = r.jobs[0];
  const part = j.set.slice(j.pos, j.pos + CHUNK);
  j.sum += r.model.loss(part) * part.length;
  j.pos += part.length;
  if (j.pos >= j.set.length) { r.jobs.shift(); j.done(j.sum / j.set.length); }
}

function writeNames(r: Run): void {
  const rr = rng(5);
  r.samples = Array.from({ length: 6 }, () => r.model.sample(rr) || '·');
  r.samplesAt = r.iter;
}

function copyChunk(r: Run): void {
  const c = r.copy ??= { train: 0, val: 0, fresh: 0, n: 0, examples: [], rand: rng(9) };
  const { trainSet, valSet } = S();
  for (let i = 0; i < 5 && c.n < COPY_N; i++) {
    const s = r.model.sample(c.rand);
    c.n++;
    if (trainSet.has(s)) c.train++;
    else if (valSet.has(s)) c.val++;
    else { c.fresh++; if (c.examples.length < 6 && s.length >= 3) c.examples.push(s); }
  }
}

// how far the run should have trained at each step
const PLAN = [0, 0, 50, WARM, 300, 600, TOTAL, TOTAL, TOTAL, TOTAL];
const COPY_STEP = 7, OWN_STEP = 8;

const f3 = (v: number | undefined) => (v === undefined ? '—' : Number.isFinite(v) ? v.toFixed(3) : Number.isNaN(v) ? 'NaN' : '∞');
const sci = (v: number) => { if (!v) return '0'; const e = Math.floor(Math.log10(v)); return `${(v / 10 ** e).toFixed(2)} × 10^${e}`; };
const pct = (v: number) => Math.round(v * 100) + '%';
const n0 = (v: number) => v.toLocaleString('en-US');

export const trainLive: Lesson = {
  id: 'train-live',
  title: { en: 'Train It Live', zh: '現場訓練它' },
  blurb: {
    en: 'Train the whole tiny GPT from random numbers, right here, and watch every curve and every attention weight change.',
    zh: '讓整個迷你 GPT 從隨機數字開始，就在這裡訓練，看著每一條曲線、每一個注意力權重如何改變。',
  },
  minutes: 12,
  level: 'intermediate',
  learn: [
    { text: { en: 'The full recipe: Adam, warmup and decay', zh: '完整的配方：Adam、暖身與衰減' }, step: 1 },
    { text: { en: 'Why we track a held-out loss too', zh: '為什麼還要追蹤驗證損失' }, step: 2 },
    { text: { en: 'How names and attention improve over a run', zh: '名字與注意力如何隨訓練進步' }, step: 5 },
    { text: { en: 'How to check that the model is not just copying', zh: '如何確認模型不是只會照抄' }, step: 7 },
    { text: { en: 'How the learning rate can make or break a run', zh: '學習率如何成就或搞砸一次訓練' }, step: 8 },
  ],
  steps: [
    {
      short: { en: 'Random', zh: '隨機' }, chapter: 'problem',
      title: { en: 'The model from last lesson, but empty', zh: '上一課的模型，但腦袋空空' },
      body: [
        { en: 'This is the exact model you assembled in Build a Tiny GPT, with all 3,115 weights set to random numbers. It writes the gibberish at the bottom.', zh: '這就是你在「打造一個迷你 GPT」組裝好的那個模型，只是 3,115 個權重全都是隨機數字。它寫出來的就是下面那些亂碼。' },
        { en: `On the right is its attention grid for <code>${WORD}</code>: who each letter looks at. With random weights it spreads its attention almost evenly. Nothing is learned yet.`, zh: `右邊是它讀 <code>${WORD}</code> 時的注意力格子：每個字母在看誰。權重是隨機的，所以注意力幾乎平均攤開。它還什麼都沒學到。` },
      ],
      callout: { kind: 'insight', text: { en: 'Everything in this lesson is computed right now in your browser. Nothing is pre-recorded.', zh: '這一課的所有東西，都是此刻在你的瀏覽器裡算出來的，沒有任何預先錄好的結果。' } },
    },
    {
      short: { en: 'Recipe', zh: '配方' }, chapter: 'technique',
      title: { en: 'The recipe: Adam, warmup, then decay', zh: '配方：Adam、暖身，然後衰減' },
      body: [
        { en: `We train with <b>Adam</b> (Optimizers lesson) on batches of ${BATCH} names for ${n0(TOTAL)} iterations. The chart shows the planned learning rate for the whole run.`, zh: `我們用 <b>Adam</b>（「最佳化器」那一課）訓練，每個批次 ${BATCH} 個名字，總共 ${n0(TOTAL)} 次迭代。圖中是整次訓練預定的學習率。` },
        { en: `It climbs from almost 0 to ${PEAK} over the first ${WARM} iterations (<b>warmup</b>), then glides down a cosine curve to a tenth of that (Learning Rate and Warmup lesson).`, zh: `前 ${WARM} 次迭代從接近 0 爬升到 ${PEAK}（<b>暖身</b>），之後沿著餘弦曲線慢慢降到它的十分之一（「學習率與暖身」那一課）。` },
      ],
      callout: { kind: 'note', text: { en: 'Warmup protects the random model from huge early steps. Decay lets it settle into fine details at the end.', zh: '暖身保護隨機的模型，不讓它一開始就跨出太大的步伐；衰減則讓它在最後能細細調整。' } },
      changed: [{ sign: '+', text: { en: 'Learning rate schedule', zh: '學習率排程' } }],
    },
    {
      short: { en: 'Two curves', zh: '兩條曲線' }, chapter: 'technique',
      title: { en: 'Watch two losses, not one', zh: '看兩條損失曲線，而不是一條' },
      body: [
        { en: 'The teal line is the <b>training loss</b>: how wrong the model is on the batches it learns from. The amber line is the <b>held-out loss</b>, measured on names it never trains on.', zh: '青綠色的線是<b>訓練損失</b>：模型在它拿來學習的批次上錯得多離譜。琥珀色的線是<b>驗證損失</b>，用它從沒訓練過的名字來量。' },
        { en: `Every ${EVERY} iterations we pause to measure the held-out loss on the same ${EVAL_N} held-out names (Overfitting and Validation lesson). After 50 iterations both have already dropped from the guessing line. (At first the training line sits slightly <i>above</i>: each of its points averages the last ${EVERY} batches, so it lags behind a model that is improving fast.)`, zh: `每 ${EVERY} 次迭代，我們停下來用同一批 ${EVAL_N} 個驗證集名字量一次（「過度擬合與驗證」那一課）。才 50 次迭代，兩條線都已經離開亂猜的那條線。（一開始訓練損失的線會稍微<i>高</i>一點：它的每個點是最近 ${EVERY} 個批次的平均，所以會落後於一個進步很快的模型。）` },
      ],
      changed: [
        { sign: '+', text: { en: 'Training and held-out loss', zh: '訓練損失與驗證損失' } },
        { sign: '~', text: { en: 'Iteration 50', zh: '第 50 次迭代' } },
      ],
    },
    {
      short: { en: 'Warmed up', zh: '暖身完' }, chapter: 'solution',
      title: { en: `Iteration ${WARM}: warmup is over`, zh: `第 ${WARM} 次迭代：暖身結束` },
      body: [
        { en: 'The learning rate has reached its peak. Most of the easy progress is already done: the model has learned which letters are common.', zh: '學習率已經到達最高點。大部分容易的進步已經完成：模型學會了哪些字母比較常見。' },
        { en: 'Look at the names: vowels start to appear between consonants. The attention grid is still blurry.', zh: '看看名字：子音之間開始出現母音了。注意力格子仍然很模糊。' },
      ],
      changed: [{ sign: '~', text: { en: `Iteration ${WARM}, peak learning rate`, zh: `第 ${WARM} 次迭代，學習率最高` } }],
    },
    {
      short: { en: '300', zh: '300' }, chapter: 'solution',
      title: { en: 'Iteration 300: slower, steadier gains', zh: '第 300 次迭代：進步變慢，但很穩定' },
      body: [
        { en: 'The curves have bent. Each new iteration buys less than the last, which is normal: the remaining mistakes are the hard ones.', zh: '曲線已經彎下來了。每一次新的迭代換來的進步都比上一次少，這很正常：剩下的錯誤都是難的。' },
        { en: 'The attention grid is getting structure: some cells darken while others fade. The model is learning <i>which</i> earlier letters are worth looking at.', zh: '注意力格子開始有了結構：有些格子變深，有些變淡。模型正在學「<i>哪些</i>」前面的字母值得看。' },
      ],
      changed: [{ sign: '~', text: { en: 'Iteration 300', zh: '第 300 次迭代' } }],
    },
    {
      short: { en: '600', zh: '600' }, chapter: 'solution',
      title: { en: 'Iteration 600: the learning rate comes down', zh: '第 600 次迭代：學習率往下降' },
      body: [
        { en: 'The learning rate is now well below its peak, so each step is smaller. The losses still creep down, more smoothly than before.', zh: '現在學習率已經明顯低於最高點，所以每一步都更小。損失仍在慢慢下降，而且比之前平順。' },
        { en: 'The names read more like names. Note the small gap opening between the two curves: the model does a little better on names it trains on.', zh: '名字讀起來越來越像名字了。注意兩條曲線之間開始出現一點縫隙：模型在練過的名字上表現稍微好一些。' },
      ],
      changed: [{ sign: '~', text: { en: 'Iteration 600', zh: '第 600 次迭代' } }],
    },
    {
      short: { en: 'Done', zh: '完成' }, chapter: 'solution',
      title: { en: `Iteration ${n0(TOTAL)}: a trained model`, zh: `第 ${n0(TOTAL)} 次迭代：訓練完成的模型` },
      body: [
        { en: `The run is over. The learning rate has decayed to a tenth of its peak, and the held-out loss levels off near the end; the amber dot marks the best checkpoint. The curve uses ${EVAL_N} of the 615 held-out names to stay fast, so at the end we also score all 615 (right), the number to compare with other lessons.`, zh: `訓練結束了。學習率衰減到最高點的十分之一，驗證損失到最後幾乎持平；琥珀色的點標出最好的檢查點。為了跑得快，這條曲線只用 615 個驗證集名字中的 ${EVAL_N} 個，所以最後我們也用全部 615 個評一次分（右邊），那才是能和其他課比較的數字。` },
        { en: 'Compare the attention grid with the first step. “Focus” is how much, on average, each letter puts on its favourite earlier letter. It has climbed well above where it started: the even spread is gone, and each letter now picks the earlier letters that help it predict.', zh: '把注意力格子和第一步比一比。「專注度」是每個字母平均放在它最喜歡的那個前面字母上的比例。它比一開始高出許多：平均攤開的樣子不見了，每個字母現在會挑選對預測有幫助的那幾個前面的字母。' },
      ],
      callout: { kind: 'insight', text: { en: 'Nobody programmed the attention pattern or the spelling rules. Both came out of the same loop: predict, measure the loss, nudge every weight.', zh: '沒有人寫程式去規定注意力的模式或拼字規則。兩者都來自同一個迴圈：預測、量損失、推動每一個權重。' } },
      changed: [{ sign: '~', text: { en: `Iteration ${n0(TOTAL)}`, zh: `第 ${n0(TOTAL)} 次迭代` } }],
    },
    {
      short: { en: 'Copying?', zh: '照抄？' }, chapter: 'problem',
      title: { en: 'Case: is it just copying the list?', zh: '案例：它是不是只在照抄清單？' },
      body: [
        { en: `A fair worry: maybe it just memorized the training names. So we ask it for ${COPY_N} names and check each one against the training list and the held-out list (the validation set).`, zh: `一個合理的懷疑：也許它只是把訓練用的名字背起來了。所以我們請它寫 ${COPY_N} 個名字，逐一和訓練清單、驗證集比對。` },
        { en: 'Most of what it writes is not in the training list. A few even match real names from the held-out list, which it never saw. It learned what names look like, not just the list itself. New is not the same as good, though: read the names too.', zh: '它寫出的名字大多不在訓練清單裡。有幾個甚至和驗證集裡的真實名字一模一樣，而它從沒看過那些名字。它學到的是「名字長什麼樣子」，而不只是那份清單。不過，「新」不等於「好」：名字本身也要讀一讀。' },
      ],
      callout: { kind: 'note', text: { en: 'Copying does happen when there is little data: in Overfitting and Validation a model memorized just 30 names. Then the held-out curve rises while the training curve keeps falling, the warning sign to watch for.', zh: '資料太少時，照抄真的會發生：在「過度擬合與驗證」那一課，模型就把僅僅 30 個名字背了下來。那時驗證損失會上升、訓練損失卻繼續下降，這正是要留意的警訊。' } },
      changed: [{ sign: '+', text: { en: `${COPY_N} names checked`, zh: `檢查 ${COPY_N} 個名字` } }],
    },
    {
      short: { en: 'Your turn', zh: '換你' }, chapter: 'problem',
      title: { en: 'Try it: run your own settings', zh: '動手試試：用你自己的設定訓練' },
      body: [
        { en: `Try it: set the peak learning rate, the warmup and the batch size. Every change restarts the run from the same random weights. Start with the recipe above (${PEAK}, warmup ${WARM}, batch ${BATCH}), then push the learning rate toward 0.3 and watch the losses shoot up to the top of the chart.`, zh: `動手試試：設定最高學習率、暖身和批次大小。每改一次，訓練就從同樣的隨機權重重新開始。先用上面的配方（${PEAK}、暖身 ${WARM}、批次 ${BATCH}），再把學習率往 0.3 推，看兩條損失衝到圖的最上方。` },
        { en: 'The gray line is your previous run, for comparison. In this tiny model a sensible learning rate matters far more than warmup: past about 0.1 a run becomes a gamble, and some settings blow up and never come back. Try a very small rate too: it is safe, but slow.', zh: '灰線是你上一次的訓練，方便比較。在這個小模型裡，合理的學習率比暖身重要得多：超過大約 0.1，訓練就像在賭運氣，有些設定會爆掉，而且再也回不來。也試試很小的學習率：很安全，但很慢。' },
      ],
      callout: { kind: 'note', text: { en: 'A loss of ∞ means the model gave the right letter a probability so small it rounded to 0. The next lesson reads a logged run that breaks the same way.', zh: '損失變成 ∞，代表模型給正確字母的機率小到被捨入成 0。下一課會讀一份紀錄，看一次以同樣方式壞掉的訓練。' } },
      changed: [{ sign: '+', text: { en: 'Your own learning rate, warmup and batch', zh: '你自己的學習率、暖身與批次' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'A whole training run, start to finish', zh: '一次完整的訓練，從頭到尾' },
      body: [
        { en: '<b>Recipe</b>: Adam, warmup, cosine decay.<br><b>Watch</b>: training loss and held-out loss together.<br><b>Results</b>: better names, and attention that found a pattern.<br><b>Check</b>: new names, not copies.', zh: '<b>配方</b>：Adam、暖身、餘弦衰減。<br><b>觀察</b>：訓練損失與驗證損失一起看。<br><b>成果</b>：更像樣的名字，以及找到規律的注意力。<br><b>檢查</b>：寫出的是新名字，而不是照抄。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Read a Real Training Run. A longer logged run, plus a run that breaks, and how to tell them apart from the charts alone.', zh: '下一課：讀懂一次真實的訓練。一次更長、有完整紀錄的訓練，加上一次失敗的訓練，以及如何只靠圖表分辨兩者。' } },
    },
  ],

  tick(step, input) {
    const until = performance.now() + 10;
    const target = PLAN[step];
    let r: Run;
    if (step === OWN_STEP) r = ownFor(input); // a changed setting starts a new run
    else {
      r = get();
      // never show a later moment than this step describes
      if (r.iter > target) r = fresh();
    }
    const pending = () => r.jobs.length > 0 || r.iter < target || r.samplesAt !== r.iter || (step === COPY_STEP && (r.copy?.n ?? 0) < COPY_N);
    while (pending() && performance.now() < until) {
      if (r.jobs.length) evalChunk(r);
      else if (r.iter < target) { trainOne(r); if (r.iter % 100 === 0) writeNames(r); }
      else if (r.samplesAt !== r.iter) writeNames(r);
      else copyChunk(r);
    }
    return pending();
  },

  reset() { run = null; own = null; ghost = null; },

  scene(step, input = {}) {
    // drawing never starts or swaps a run (main.ts also calls scene with no input, just to read the controls)
    const r = step === OWN_STEP ? (own && same(own.cfg, cfgOf(input)) ? own : own ?? make(cfgOf(input))) : get();
    const items: Item[] = [];
    const target = PLAN[step];
    const running = r.iter < target || r.jobs.length > 0;

    // attention grid (or the copying check)
    const grid = attGrid(r.model);
    if (step !== COPY_STEP) {
      items.push({ k: 'ah', t: 'text', x: 600, y: 34, size: 13.5, tone: 'muted', text: t({ en: `attention while reading “${WORD}”`, zh: `讀「${WORD}」時的注意力` }) });
      items.push({ k: 'heat', t: 'heat', x: 620, y: 70, m: grid, rows: LETTERS, cols: LETTERS, cell: 40, fmt: 'pct' });
      items.push({ k: 'af', t: 'text', x: 740, y: 342, anchor: 'middle', size: 15, bold: true, tone: 'a', text: t({ en: `focus ${pct(focus(grid))}`, zh: `專注度 ${pct(focus(grid))}` }) });
      items.push({ k: 'af2', t: 'text', x: 740, y: 364, anchor: 'middle', size: 13, tone: 'muted', text: t({ en: `at iteration 0: ${pct(focus(r.att0))}`, zh: `第 0 次迭代時：${pct(focus(r.att0))}` }) });
      if ((step === 6 || step === OWN_STEP) && r.valAll !== undefined) items.push({ k: 'vall', t: 'text', x: 740, y: 462, anchor: 'middle', size: 13.5, bold: true, tone: 'b', text: t({ en: `all 615 held-out names: ${f3(r.valAll)}`, zh: `全部 615 個驗證集名字：${f3(r.valAll)}` }) });
      if (step >= 1) {
        items.push({ k: 'ops', t: 'text', x: 740, y: 400, anchor: 'middle', size: 13, tone: 'muted', text: t({ en: 'operations so far ≈ 6 × 3,115 × tokens', zh: '目前的運算量 ≈ 6 × 3,115 × 詞元數' }) });
        items.push({ k: 'ops2', t: 'text', x: 740, y: 422, anchor: 'middle', size: 15, bold: true, mono: true, tone: 'a', text: sci(OPS_PER_TOKEN * r.tokens) });
      }
    } else {
      const c = r.copy;
      const n = Math.max(1, c?.n ?? 0);
      items.push({
        k: 'cb', t: 'bars', x: 590, y: 60, w: 280, h: 230,
        title: t({ en: `${c?.n ?? 0} of ${COPY_N} names written`, zh: `已寫出 ${c?.n ?? 0} / ${COPY_N} 個名字` }),
        items: [
          { label: t({ en: 'in training', zh: '訓練清單' }), v: (c?.train ?? 0) / n },
          { label: t({ en: 'held-out', zh: '驗證集' }), v: (c?.val ?? 0) / n },
          { label: t({ en: 'brand new', zh: '全新' }), v: (c?.fresh ?? 0) / n, hi: true },
        ],
      });
      items.push({ k: 'cn', t: 'text', x: 590, y: 322, size: 13.5, tone: 'muted', text: t({ en: 'some brand-new names it wrote:', zh: '它寫出的一些全新名字：' }) });
      const ex = c?.examples ?? [];
      items.push({ k: 'ce', t: 'text', x: 590, y: 348, size: 15, mono: true, bold: true, tone: 'a', text: ex.slice(0, 3).join('  ') || '…' });
      items.push({ k: 'ce2', t: 'text', x: 590, y: 372, size: 15, mono: true, bold: true, tone: 'a', text: ex.slice(3, 6).join('  ') });
    }

    // learning rate: the plan, and how far we are
    if (step >= 1) {
      const plan = Array.from({ length: TOTAL / EVERY + 1 }, (_, i) => (lrOf(r.cfg, i * EVERY) / r.cfg.peak) * 100);
      const done = plan.slice(0, Math.floor(r.iter / EVERY) + 1);
      items.push({
        k: 'lr', t: 'chart', x: 70, y: 300, w: 440, h: 80, ymin: 0, ymax: 100, xmax: TOTAL, every: EVERY,
        series: [{ pts: plan, tone: 'muted' }, { pts: done, tone: 'a' }],
        xlabel: t({ en: 'iterations', zh: '迭代次數' }), ylabel: t({ en: `learning rate, % of peak (${r.cfg.peak})`, zh: `學習率，占最高值（${r.cfg.peak}）的 %` }),
      });
    }

    // the two loss curves
    if (step >= 2) {
      items.push({
        k: 'loss', t: 'chart', x: 70, y: 50, w: 440, h: 180, ymin: 2, ymax: 3.6, xmax: TOTAL, every: EVERY,
        series: [...(step === OWN_STEP && ghost ? [{ pts: ghost.val, tone: 'muted' as const }] : []), { pts: r.train, tone: 'a' }, { pts: r.val, tone: 'b' }],
        xlabel: t({ en: 'iterations', zh: '迭代次數' }), ylabel: t({ en: 'loss (lower is better)', zh: '損失（越低越好）' }),
        guide: { y: UNIFORM, label: t({ en: 'pure guessing · 3.30', zh: '純粹亂猜 · 3.30' }) },
      });
      // the best checkpoint so far (Checkpoints and Evaluation lesson)
      const fin = r.val.map((v) => (Number.isFinite(v) ? v : Infinity));
      const bi = fin.indexOf(Math.min(...fin));
      if ((step === 6 || step === OWN_STEP) && r.val.length > 1 && Number.isFinite(fin[bi])) {
        const bx = 70 + ((bi * EVERY) / TOTAL) * 440, by = 50 + 180 - ((Math.min(3.6, Math.max(2, fin[bi])) - 2) / 1.6) * 180;
        items.push({ k: 'best', t: 'dot', x: bx, y: by, r: 6, tone: 'b' });
        items.push({ k: 'bestT', t: 'text', x: Math.min(bx, 500), y: by - 12, anchor: 'end', size: 12.5, bold: true, tone: 'b', text: t({ en: `best ${fin[bi].toFixed(3)} @${bi * EVERY}`, zh: `最佳 ${fin[bi].toFixed(3)} @${bi * EVERY}` }) });
      }
      if (step === OWN_STEP && ghost) items.push({ k: 'ghl', t: 'text', x: 510, y: 44, anchor: 'end', size: 12.5, tone: 'muted', text: t({ en: `gray: previous run (${ghost.cfg.peak}, warmup ${ghost.cfg.warm}, batch ${ghost.cfg.batch})`, zh: `灰線：上一次（${ghost.cfg.peak}、暖身 ${ghost.cfg.warm}、批次 ${ghost.cfg.batch}）` }) });
    } else {
      const v = r.val[0];
      items.push({ k: 'big', t: 'text', x: 290, y: 150, anchor: 'middle', size: 30, bold: true, tone: 'bad', text: f3(v) });
      items.push({ k: 'bigl', t: 'text', x: 290, y: 180, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: `held-out loss · pure guessing is ${UNIFORM.toFixed(2)}`, zh: `驗證損失 · 純粹亂猜是 ${UNIFORM.toFixed(2)}` }) });
    }

    // what it writes
    items.push({ k: 'nl', t: 'text', x: 70, y: 432, size: 14, tone: 'muted', text: t({ en: `the model writes (iteration ${n0(r.samplesAt < 0 ? 0 : r.samplesAt)}):`, zh: `模型寫出（第 ${n0(r.samplesAt < 0 ? 0 : r.samplesAt)} 次迭代）：` }) });
    (r.samples.length ? r.samples : Array(6).fill('…')).forEach((s, i) => items.push({ k: 'n' + i, t: 'box', x: 70 + i * 92, y: 444, w: 84, h: 40, label: s, active: running }));

    const last = (a: number[]) => a[a.length - 1];
    return {
      metrics: [
        { label: { en: 'Iteration', zh: '迭代次數' }, value: `${n0(r.iter)} / ${n0(TOTAL)}` },
        { label: { en: 'Learning rate', zh: '學習率' }, value: step >= 1 ? lrOf(r.cfg, Math.min(r.iter, TOTAL - 1)).toFixed(4) : '—' },
        { label: { en: 'Training loss', zh: '訓練損失' }, value: f3(last(r.train)) },
        { label: { en: 'Held-out loss', zh: '驗證損失' }, value: f3(last(r.val)) },
      ],
      controls: step === OWN_STEP ? [
        { key: 'lr', kind: 'slider', label: { en: 'Peak learning rate', zh: '最高學習率' }, min: -3, max: -0.5, step: 0.05, value: +Math.log10(PEAK).toFixed(2), format: (v) => String(+(10 ** v).toPrecision(2)) },
        { key: 'warm', kind: 'choice', label: { en: 'Warmup', zh: '暖身' }, value: String(WARM), options: ['0', '100', '300'].map((v) => ({ value: v, label: { en: v, zh: v } })) },
        { key: 'batch', kind: 'choice', label: { en: 'Batch', zh: '批次' }, value: String(BATCH), options: ['8', '16', '32'].map((v) => ({ value: v, label: { en: v, zh: v } })) },
      ] : undefined,
      items,
      legend: step >= 2
        ? [{ tone: 'a', text: { en: 'training loss', zh: '訓練損失' } }, { tone: 'b', text: { en: 'held-out loss', zh: '驗證損失' } }]
        : undefined,
    };
  },
};

