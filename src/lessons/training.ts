import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t } from '../i18n';
import { GPT, DEFAULT_CONFIG, sgdStep } from '../model/gpt';
import { CHARS, TRAIN_NAMES, VAL_NAMES, example, rng } from '../model/data';

// This lesson really trains the model in the browser, starting from random weights.

const ALL = TRAIN_NAMES.map(example);
const EVAL = VAL_NAMES.slice(0, 150).map(example); // fixed held-out names used to measure the loss
const BATCH = 16;
const EVERY = 10; // measure the loss every 10 iterations
const UNIFORM = Math.log(27);

interface Run { model: GPT; rand: () => number; lr: number; iter: number; hist: number[]; samples: string[]; broke: boolean }
const LR = { main: 0.5, hot: 4, cold: 0.01 } as const;
type RunName = keyof typeof LR | 'try';
const runs = new Map<RunName, Run>();
// learning rates the "try it" slider steps through (slider value = index)
const TRY_LRS = [0.01, 0.03, 0.1, 0.3, 0.5, 1, 2, 4];
const TRY_START = 4; // 0.5, the rate of the main run
const tryLr = (input?: Record<string, unknown>) => TRY_LRS[Math.max(0, Math.min(TRY_LRS.length - 1, Math.round(Number(input?.lri ?? TRY_START)) || 0))];

const writeNames = (model: GPT) => { const r = rng(5); return Array.from({ length: 6 }, () => model.sample(r) || '·'); };

function fresh(name: RunName, lr: number = name === 'try' ? LR.main : LR[name]): Run {
  const model = new GPT(DEFAULT_CONFIG, rng(1));
  const run: Run = { model, rand: rng(2), lr, iter: 0, hist: [START_LOSS()], samples: writeNames(model), broke: false };
  runs.set(name, run);
  return run;
}
const get = (name: RunName) => runs.get(name) ?? fresh(name);
// every run starts from the same random weights, so its starting loss is the same
let startLoss: number | undefined;
function START_LOSS(): number { return (startLoss ??= new GPT(DEFAULT_CONFIG, rng(1)).loss(EVAL)); }
/** For drawing only: the "try it" run if it uses this rate, else what a new one looks like (never replaces the live run). */
const peekTry = (lr: number): Run => {
  const r = runs.get('try');
  if (r && r.lr === lr) return r;
  const model = new GPT(DEFAULT_CONFIG, rng(1));
  return { model, rand: rng(2), lr, iter: 0, hist: [START_LOSS()], samples: writeNames(model), broke: false };
};
/** The "try it" run for this learning rate; a different rate starts it again from the same random start. */
const getTry = (lr: number) => { const r = runs.get('try'); return r && r.lr === lr ? r : fresh('try', lr); };

function trainOne(run: Run): void {
  const batch = Array.from({ length: BATCH }, () => ALL[Math.floor(run.rand() * ALL.length)]);
  run.model.loss(batch, true);
  sgdStep(run.model, run.lr);
  run.iter++;
  if (run.iter % EVERY === 0) {
    const l = run.model.loss(EVAL);
    run.hist.push(l);
    if (!Number.isFinite(l)) run.broke = true;
  }
  if (run.iter % 100 === 0 && !run.broke) run.samples = writeNames(run.model);
}

// which runs each step needs, and how far each should have trained
const PLAN: [RunName, number][][] = [
  [['main', 0]], [['main', 0]], [['main', 0]], [['main', 0]],
  [['main', 300]], [['main', 1500]],
  [['main', 300], ['hot', 300]], [['main', 300], ['cold', 300]],
  [['main', 300], ['try', 300]],
  [['main', 1500]],
];
const TRY = 8, RECAP = 9;

// Numbers for the first steps, computed once from the untrained model.
const intro = (() => {
  const model = new GPT(DEFAULT_CONFIG, rng(1));
  const rand = rng(2);
  const probs = model.next([0, 5, 13, 13]);
  const before = model.loss(EVAL);
  const batch = Array.from({ length: BATCH }, () => ALL[Math.floor(rand() * ALL.length)]);
  model.loss(batch, true);
  const M = CHARS.indexOf('m');
  const w = model.p.tok.row(M);
  const g = Array.from(model.p.tok.g.subarray(M * 16, (M + 1) * 16));
  sgdStep(model, LR.main);
  const w2 = model.p.tok.row(M);
  const top = g.reduce((b, v, i) => (Math.abs(v) > Math.abs(g[b]) ? i : b), 0); // the cell with the largest gradient
  const maxW = Math.max(...w.map(Math.abs)), maxG = Math.max(...g.map(Math.abs));
  return { probs, before, after: model.loss(EVAL), w, g, w2, top, maxW, maxG, count: model.paramCount() };
})();
const sg2 = (v: number, d = 3) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
const LN = [1, 0.5, 0.1, 0.01].map((p) => `p = ${p} → ${(-Math.log(p) + 0).toFixed(2)}`).join('  ·  ');

const f3 = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : 'NaN');
const last = (r: Run) => r.hist[r.hist.length - 1];

export const training: Lesson = {
  id: 'training-loop',
  title: { en: 'The Training Loop', zh: '訓練迴圈' },
  blurb: {
    en: 'Watch a model go from random letters to plausible names, one small correction at a time.',
    zh: '看一個模型如何從亂碼變成像樣的名字，每次只修正一點點。',
  },
  minutes: 12,
  level: 'beginner',
  learn: [
    { text: { en: 'What “loss” measures', zh: '「損失」衡量的是什麼' }, step: 1 },
    { text: { en: 'What a gradient tells each weight', zh: '梯度告訴每個權重什麼' }, step: 2 },
    { text: { en: 'The loop: batch, forward, loss, backward, update', zh: '迴圈：批次、前向、損失、反向、更新' }, step: 4 },
    { text: { en: 'How the learning rate can break training', zh: '學習率如何讓訓練失敗' }, step: 6 },
  ],
  steps: [
    {
      short: { en: 'Random', zh: '隨機' }, chapter: 'problem',
      title: { en: 'A new model writes nonsense', zh: '剛出生的模型只會亂寫' },
      body: [
        { en: `Our model has ${intro.count.toLocaleString()} <b>weights</b>: the numbers inside its matrices. At the start every one is random.`, zh: `我們的模型有 ${intro.count.toLocaleString()} 個<b>權重</b>，也就是它那些矩陣裡的數字。一開始，每一個都是隨機的。` },
        { en: 'Ask it for names and you get these. The whole structure of the transformer is already there. What is missing is the right numbers.', zh: '請它寫名字，就會得到這些東西。Transformer 的結構其實已經完整了，缺的只是正確的數字。' },
      ],
      callout: { kind: 'insight', text: { en: 'Training means finding good values for the weights. Nobody sets them by hand; the model adjusts them itself by practising on examples.', zh: '訓練，就是替權重找到好的數值。沒有人動手去設定它們，而是模型靠著練習範例自己調整。' } },
    },
    {
      short: { en: 'Loss', zh: '損失' }, chapter: 'technique',
      title: { en: 'Measure how wrong it is', zh: '量出它錯得多離譜' },
      body: [
        { en: 'Give the model <code>.emm</code>. The right next letter is “a”. The chart shows the probability it gives every letter; “a” is highlighted.', zh: '給模型 <code>.emm</code>，正確的下一個字母是「a」。圖中是它給每個字母的機率，「a」被標了出來。' },
        { en: 'The <b>loss</b> is −ln(probability of the right answer). A confident right answer gives a loss near 0; a low probability gives a large loss. Averaged over many examples it is one number for “how wrong is the model”.', zh: '<b>損失（loss）</b>就是 −ln(正確答案的機率)。很有把握地答對，損失接近 0；給正確答案的機率越低，損失越大。對很多範例取平均，就得到一個代表「模型錯得多離譜」的數字。' },
      ],
      callout: { kind: 'note', text: { en: 'The “held-out loss” tile is the average over 150 held-out names: names the model never trains on, kept aside to check it honestly. That is why it differs from this one example. Guessing all 27 tokens equally gives ln 27 ≈ 3.30; the untrained model is a little worse than that.', zh: '上方「驗證損失」是 150 個驗證集名字的平均損失：驗證集是模型訓練時從沒看過、特地留下來公正檢查它的名字。所以它和這一個例子的損失不一樣。如果 27 個詞元機率均等地亂猜，損失是 ln 27 ≈ 3.30；未經訓練的模型比這還差一點。' } },
      changed: [{ sign: '+', text: { en: 'A single number to push down', zh: '一個要壓低的數字' } }],
    },
    {
      short: { en: 'Gradient', zh: '梯度' }, chapter: 'technique',
      title: { en: 'Ask every weight which way to move', zh: '問每個權重該往哪邊動' },
      body: [
        { en: 'For each weight we can compute a <b>gradient</b>: if this weight grew a tiny bit, would the loss go up or down, and how fast?', zh: '對每個權重，我們都能算出一個<b>梯度（gradient）</b>：如果這個權重增加一點點，損失會上升還是下降？變化有多快？' },
        { en: `The top strip shows 16 of the model’s weights (the vector for “m”). The strip below is their gradients, about ${Math.round(intro.maxW / intro.maxG)}× smaller, so it gets its own colour scale. All ${intro.count.toLocaleString()} weights get a gradient, computed in a single backward pass through the model.`, zh: `上面那一條是模型的其中 16 個權重（「m」的向量），下面那一條是它們的梯度，大約小了 ${Math.round(intro.maxW / intro.maxG)} 倍，所以用它自己的顏色刻度。全部 ${intro.count.toLocaleString()} 個權重都會得到一個梯度，只要從模型的尾端往回算一遍就能全部求出。` },
      ],
      callout: { kind: 'note', text: { en: 'That backward pass is called backpropagation. The Gradients lesson opens it up.', zh: '這個往回算的過程叫做反向傳播（backpropagation），在「梯度」那一課會拆開來看。' } },
      changed: [{ sign: '+', text: { en: 'One gradient per weight', zh: '每個權重一個梯度' } }],
    },
    {
      short: { en: 'One step', zh: '走一步' }, chapter: 'technique',
      title: { en: 'Nudge every weight downhill', zh: '把每個權重往下坡推一點' },
      body: [
        { en: `Now move every weight a little <i>against</i> its gradient: <code>weight = weight − learning rate × gradient</code>. The framed cell is worked out at the bottom: ${intro.w[intro.top].toFixed(3)} − ${LR.main} × (${sg2(intro.g[intro.top], 4)}) = ${intro.w2[intro.top].toFixed(3)}.`, zh: `現在把每個權重往梯度的<i>反方向</i>移動一點：<code>權重 = 權重 − 學習率 × 梯度</code>。框起來的那一格，算式寫在最下面：${intro.w[intro.top].toFixed(3)} − ${LR.main} × (${sg2(intro.g[intro.top], 4)}) = ${intro.w2[intro.top].toFixed(3)}。` },
        { en: `The <b>learning rate</b> sets the size of the nudge; here it is ${LR.main}. After just one step the loss drops from ${f3(intro.before)} to ${f3(intro.after)}.`, zh: `<b>學習率（learning rate）</b>決定每次推多遠，這裡是 ${LR.main}。只走了一步，損失就從 ${f3(intro.before)} 降到 ${f3(intro.after)}。` },
      ],
      changed: [
        { sign: '~', text: { en: `All ${intro.count.toLocaleString()} weights updated`, zh: `${intro.count.toLocaleString()} 個權重全部更新` } },
        { sign: '~', text: { en: `Loss ${f3(intro.before)} → ${f3(intro.after)}`, zh: `損失 ${f3(intro.before)} → ${f3(intro.after)}` } },
      ],
    },
    {
      short: { en: 'The loop', zh: '迴圈' }, chapter: 'solution',
      title: { en: 'Do it again, 300 times', zh: '再做一次，重複 300 次' },
      body: [
        { en: `One step is a start, but the loss is still near guessing, so we loop. Each <b>iteration</b> picks a <b>batch</b> of ${BATCH} random names, runs the model forward, measures the loss, runs backward for gradients, and updates the weights.`, zh: `走一步是個開始，但損失仍接近亂猜，所以要不斷重複。每一次<b>迭代（iteration）</b>都隨機挑 ${BATCH} 個名字組成一個<b>批次（batch）</b>，讓模型前向計算、量出損失、反向算出梯度，然後更新權重。` },
        { en: 'This is running for real in your browser right now. Watch the loss fall and the names change.', zh: '這個過程此刻正在你的瀏覽器裡真實執行。看著損失下降、名字改變吧。' },
      ],
      changed: [{ sign: '+', text: { en: 'Training loop, 300 iterations', zh: '訓練迴圈，300 次迭代' } }],
    },
    {
      short: { en: 'Keep going', zh: '繼續練' }, chapter: 'solution',
      title: { en: 'From noise to names', zh: '從雜訊到名字' },
      body: [
        { en: 'Keep looping to 1,500 iterations. The loss falls quickly at first, then more and more slowly.', zh: '繼續跑到 1,500 次迭代。損失一開始掉得很快，之後越來越慢。' },
        { en: 'Look at what the model writes now. Nobody told it that names mix vowels and consonants, or roughly how long a name is. Whatever it has picked up, it learned only by being pushed, step after step, to predict the next letter a little better.', zh: '看看模型現在寫出的東西。沒有人告訴它名字要母音、子音交錯，或一個名字大概多長。它學到的一切，都只是一步一步被推著把下一個字母猜得更準一點而來的。' },
      ],
      callout: { kind: 'insight', text: { en: 'ChatGPT-scale models are trained by this same loop. The differences are scale: billions of weights, trillions of tokens, and weeks on thousands of chips.', zh: 'ChatGPT 等級的模型，用的也是同一個迴圈。差別只在規模：數十億個權重、數兆個詞元，以及在數千顆晶片上跑好幾週。' } },
      changed: [{ sign: '~', text: { en: '1,500 iterations', zh: '1,500 次迭代' } }],
    },
    {
      short: { en: 'Too fast', zh: '太快' }, chapter: 'problem',
      title: { en: 'Case: the learning rate is too high', zh: '案例：學習率太高' },
      body: [
        { en: `Impatient, we raise the learning rate from ${LR.main} to ${LR.hot} so each step moves further. The red line is this new run; the grey line is the earlier one.`, zh: `我們沒耐心，把學習率從 ${LR.main} 調到 ${LR.hot}，讓每一步跨得更遠。紅線是這次的新訓練，灰線是先前的那一次。` },
        { en: 'The steps overshoot. The weights fly off to enormous values, the loss stops being a number at all (NaN), and the model is ruined.', zh: '步伐跨過頭了。權重衝到極大的數值，損失甚至不再是一個數字（NaN），整個模型就毀了。' },
      ],
      callout: { kind: 'note', text: { en: 'A gradient only describes the slope right where you stand. Jump too far and you land somewhere it never described.', zh: '梯度只描述你腳下這一小塊的坡度。跳得太遠，就會落在它從沒描述過的地方。' } },
      changed: [
        { sign: '~', text: { en: `Learning rate ${LR.main} → ${LR.hot}`, zh: `學習率 ${LR.main} → ${LR.hot}` } },
        { sign: '~', text: { en: 'Training diverges', zh: '訓練發散' } },
      ],
    },
    {
      short: { en: 'Too slow', zh: '太慢' }, chapter: 'problem',
      title: { en: 'Case: the learning rate is too low', zh: '案例：學習率太低' },
      body: [
        { en: `So we play safe and set it to ${LR.cold}. Nothing breaks, but look at the red line: after 300 iterations it has hardly moved.`, zh: `那就保守一點，設成 ${LR.cold}。沒有東西壞掉，但是看看紅線：跑了 300 次迭代，幾乎沒有動。` },
        { en: 'It would get there eventually, at many times the cost. For a large model that is the difference between days and months.', zh: '它最終還是會到，只是要花好幾倍的成本。對大型模型來說，這就是幾天和幾個月的差別。' },
      ],
      callout: { kind: 'insight', text: { en: 'The learning rate is the most important setting in training. Too high breaks it, too low wastes it. Later lessons cover warmup and schedules that change it during the run.', zh: '學習率是訓練中最重要的設定。太高會弄壞，太低是浪費。後面的課程會介紹在訓練過程中調整它的暖身（warmup）與排程。' } },
      changed: [{ sign: '~', text: { en: `Learning rate ${LR.main} → ${LR.cold}`, zh: `學習率 ${LR.main} → ${LR.cold}` } }],
    },
    {
      short: { en: 'Try it', zh: '動手試試' }, chapter: 'problem',
      title: { en: 'Try it: find where training breaks', zh: '動手試試：找出訓練在哪裡壞掉' },
      body: [
        { en: `Try it: drag the learning rate and watch the red line. Each time you move it, a new run starts from the same random start and trains 300 iterations live; the grey line is the ${LR.main} run from before.`, zh: `動手試試：拖動學習率，看看紅線。每移動一次，就從同一個隨機起點重新開始，即時訓練 300 次迭代；灰線是先前學習率 ${LR.main} 的那次訓練。` },
        { en: 'Look for the edge: somewhere between the values that work and the one that broke, the loss starts to jump or the run fails. At the small end, notice how slowly it crawls.', zh: '找找看那條邊界：在能訓練的數值和會壞掉的數值之間，損失開始亂跳，或整個訓練失敗。在最小的那一端，注意它爬得有多慢。' },
      ],
      changed: [{ sign: '+', text: { en: 'A learning-rate slider that retrains live', zh: '可即時重新訓練的學習率滑桿' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Training is a loop of small corrections', zh: '訓練就是不斷的小修正' },
      body: [
        { en: '<b>Loss</b>: one number for how wrong the model is.<br><b>Gradient</b>: which way each weight should move.<br><b>Update</b>: move a little, by the learning rate.<br><b>Loop</b>: repeat on a fresh batch, thousands of times.', zh: '<b>損失</b>：用一個數字表示模型錯得多離譜。<br><b>梯度</b>：每個權重該往哪邊動。<br><b>更新</b>：依學習率移動一點點。<br><b>迴圈</b>：換一個新批次再來一次，重複數千次。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Gradients and Backpropagation. How the model works out a gradient for every weight in one backward pass.', zh: '下一課：梯度與反向傳播。模型如何只靠一次反向計算，就求出每個權重的梯度。' } },
    },
  ],

  tick(step, input) {
    let more = false;
    const until = performance.now() + 10;
    for (const [name, target] of PLAN[step]) {
      let run = name === 'try' ? getTry(tryLr(input)) : get(name);
      // never show a later moment than this step describes (the final recap keeps whatever exists)
      if (run.iter > target && !(name === 'main' && step >= 5)) run = fresh(name, run.lr);
      while (run.iter < target && !run.broke && performance.now() < until) trainOne(run);
      if (run.iter < target && !run.broke) more = true;
    }
    return more;
  },

  reset() { runs.clear(); },

  scene(step, input = {}) {
    const items: Item[] = [];
    const main = get('main');
    const sideName: RunName | null = step === 6 ? 'hot' : step === 7 ? 'cold' : step === TRY ? 'try' : null;
    const side = sideName === 'try' ? peekTry(tryLr(input)) : sideName ? get(sideName) : null;
    const shown = side ?? main;

    const names = (run: Run, y: number, x0: number) => {
      items.push({ k: 'nl', t: 'text', x: x0, y: y - 12, text: t({ en: 'the model writes:', zh: '模型寫出：' }), size: 14.5, tone: 'muted' });
      run.samples.forEach((s, i) => items.push({ k: 'n' + i, t: 'box', x: x0 + i * 92, y, w: 84, h: 40, label: run.broke ? '???' : s, tone: run.broke ? 'bad' : 'ink' }));
    };

    if (step === 0) names(main, 230, 176);

    if (step === 1) {
      ['.', 'e', 'm', 'm'].forEach((ch, i) => items.push({ k: 'tok' + i, t: 'token', x: 300 + i * 62, y: 40, text: ch, tone: ch === '.' ? 'b' : 'ink' }));
      items.push({ k: 'tokq', t: 'token', x: 300 + 4 * 62, y: 40, text: 'a', tone: 'a', sub: t({ en: 'correct', zh: '正解' }) });
      const p = intro.probs;
      items.push({ k: 'pb', t: 'bars', x: 40, y: 170, w: 820, h: 220, items: p.map((v, i) => ({ label: CHARS[i], v, hi: i === 1 })), title: t({ en: 'probability the untrained model gives each next letter', zh: '未訓練的模型給每個「下一個字母」的機率' }) });
      items.push({ k: 'pl', t: 'text', x: 450, y: 428, anchor: 'middle', mono: true, bold: true, size: 17, tone: 'a', text: `loss = −ln(${p[1].toFixed(3)}) = ${(-Math.log(p[1])).toFixed(2)}` });
      items.push({ k: 'pln', t: 'text', x: 450, y: 466, anchor: 'middle', mono: true, size: 14.5, text: t({ en: `loss for probability ${LN}`, zh: `機率與損失：${LN}` }) });
      items.push({ k: 'plc', t: 'text', x: 450, y: 494, anchor: 'middle', size: 13.5, tone: 'muted', text: t({ en: 'ln is the natural logarithm: it turns a small probability into a big penalty', zh: 'ln 是自然對數：它把「機率很小」變成「懲罰很大」' }) });
    }

    if (step === 2 || step === 3) {
      const x0 = 250, cell = 25;
      const ratio = Math.round(intro.maxW / intro.maxG);
      const Y = step === 2 ? [110, 220] : [60, 150, 240, 330];
      items.push({ k: 'gw', t: 'vec', x: x0, y: Y[0], v: intro.w, cell, max: intro.maxW, label: t({ en: '16 weights (the vector for “m”)', zh: '16 個權重（「m」的向量）' }) });
      items.push({ k: 'gg', t: 'vec', x: x0, y: Y[1], v: intro.g, cell, max: intro.maxG, label: t({ en: `their gradients (colour scale ×${ratio}): orange = raising it raises the loss`, zh: `它們的梯度（顏色刻度放大 ${ratio} 倍）：橘色 = 調高它會讓損失上升` }) });
      if (step === 2) items.push({ k: 'ga', t: 'arrow', x1: x0 + 200, y1: 142, x2: x0 + 200, y2: 190 });
      if (step === 3) {
        const change = intro.w2.map((v, i) => v - intro.w[i]);
        items.push({ k: 'gc', t: 'vec', x: x0, y: Y[2], v: change, cell, max: Math.max(...change.map(Math.abs)), label: t({ en: `the change: −${LR.main} × gradient (its own colour scale)`, zh: `改變量：−${LR.main} × 梯度（用它自己的顏色刻度）` }) });
        items.push({ k: 'gu', t: 'vec', x: x0, y: Y[3], v: intro.w2, cell, max: intro.maxW, label: t({ en: `after the step: weight + change (same scale as the weights)`, zh: `更新之後：權重 + 改變量（和權重同一個刻度）` }) });
        const cx = x0 + intro.top * cell - 4;
        Y.forEach((y, i) => items.push({ k: 'gfr' + i, t: 'path', x: 0, y: 0, pts: [[cx, y - 4], [cx + cell + 3, y - 4], [cx + cell + 3, y + 26], [cx, y + 26]], closed: true, tone: 'b', width: 2.5 }));
        items.push({ k: 'gn', t: 'text', x: 450, y: 410, anchor: 'middle', size: 16, mono: true, bold: true, tone: 'b', text: t({
          en: `framed cell: ${intro.w[intro.top].toFixed(3)} − ${LR.main} × (${sg2(intro.g[intro.top], 4)}) = ${intro.w2[intro.top].toFixed(3)}`,
          zh: `框起來的格子：${intro.w[intro.top].toFixed(3)} − ${LR.main} × (${sg2(intro.g[intro.top], 4)}) = ${intro.w2[intro.top].toFixed(3)}` }) });
        items.push({ k: 'gn2', t: 'text', x: 450, y: 445, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: 'each change is tiny compared with the weight: training is many small steps', zh: '每個改變量和權重相比都很小：訓練就是走很多小步' }) });
      } else {
        items.push({ k: 'gn', t: 'text', x: 450, y: 320, anchor: 'middle', size: 14.5, tone: 'muted', text: t({ en: `…and the same for the other ${(intro.count - 16).toLocaleString()} weights`, zh: `……其他 ${(intro.count - 16).toLocaleString()} 個權重也一樣` }) });
      }
    }

    if (step >= 4) {
      const loop = [
        { en: `batch of ${BATCH} names`, zh: `${BATCH} 個名字的批次` }, { en: 'forward: predict', zh: '前向：做預測' }, { en: 'loss: how wrong', zh: '損失：錯多少' },
        { en: 'backward: gradients', zh: '反向：算梯度' }, { en: 'update weights', zh: '更新權重' },
      ];
      const running = shown.iter < (PLAN[step].find(([n]) => n === (sideName ?? 'main'))?.[1] ?? 0) && !shown.broke;
      loop.forEach((b, i) => {
        items.push({ k: 'lb' + i, t: 'box', x: 30, y: 40 + i * 66, w: 190, h: 44, label: t(b), active: running });
        if (i < 4) items.push({ k: 'la' + i, t: 'arrow', x1: 125, y1: 86 + i * 66, x2: 125, y2: 104 + i * 66 });
      });
      items.push({ k: 'lr', t: 'text', x: 125, y: 376, anchor: 'middle', size: 14.5, tone: 'muted', text: t({ en: '↻ then again, with a new batch', zh: '↻ 換一個新批次再來一次' }) });

      const xmax = step === 5 || step === RECAP ? 1500 : 300;
      const cut = (r: Run) => r.hist.slice(0, xmax / EVERY + 1);
      const series: { pts: number[]; tone: 'muted' | 'bad' | 'a' }[] = side
        ? [{ pts: cut(main), tone: 'muted' }, { pts: cut(side), tone: 'bad' }]
        : [{ pts: cut(main), tone: 'a' }];
      items.push({
        k: 'chart', t: 'chart', x: 320, y: 50, w: 530, h: 260, ymin: 1.5, ymax: 4, xmax, every: EVERY, series,
        xlabel: t({ en: 'iterations', zh: '迭代次數' }), ylabel: t({ en: 'loss (lower is better)', zh: '損失（越低越好）' }),
        guide: { y: UNIFORM, label: t({ en: 'pure guessing · 3.30', zh: '純粹亂猜 · 3.30' }) },
      });
      names(shown, 420, 320);
    }

    return {
      metrics: [
        { label: { en: 'Iterations', zh: '迭代次數' }, value: step === 3 ? '1' : shown.iter.toLocaleString() },
        { label: { en: 'Held-out loss', zh: '驗證損失' }, value: step === 3 ? f3(intro.after) : f3(last(shown)) },
        { label: { en: 'Learning rate', zh: '學習率' }, value: step >= 3 ? String(shown.lr) : '—' },
        { label: { en: 'Weights', zh: '權重數' }, value: intro.count.toLocaleString() },
      ],
      items,
      legend: step === 2 || step === 3
        ? [{ tone: 'pos', text: { en: 'positive', zh: '正數' } }, { tone: 'neg', text: { en: 'negative', zh: '負數' } }]
        : side
          ? [{ tone: 'bad', text: { en: `this run · learning rate ${side.lr}`, zh: `這次訓練 · 學習率 ${side.lr}` } }]
          : undefined,
      controls: step === TRY ? [{
        key: 'lri', kind: 'slider', label: { en: 'Learning rate', zh: '學習率' }, min: 0, max: TRY_LRS.length - 1, step: 1, value: TRY_START,
        format: (v: number) => String(TRY_LRS[Math.round(v)] ?? ''),
      }] : undefined,
    };
  },
};
