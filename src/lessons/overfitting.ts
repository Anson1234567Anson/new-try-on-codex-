import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t } from '../i18n';
import { GPT, DEFAULT_CONFIG } from '../model/gpt';
import { rng } from '../model/data';
import exp from '../model/experiments.json';

// Real runs from experiments.json: `overfit` trains on only 30 names; `main` is the full run.

const OF = exp.overfit;
const MAIN = exp.main;
const SPLIT = exp.split;
const num = (a: (number | null)[]) => a.map((v) => (v === null ? NaN : v));
const ofTrain = num(OF.train), ofVal = num(OF.val);
const mTrain = num(MAIN.train), mVal = num(MAIN.val);
const lastOf = (a: number[]) => a[a.length - 1];
const OF_STEPS = (ofTrain.length - 1) * OF.every;
const M_STEPS = (mTrain.length - 1) * MAIN.every;
const OF_NAMES = OF.names.length;
const bestIdx = ofVal.indexOf(Math.min(...ofVal));
const BEST_STEP = bestIdx * OF.every;
const SAMPLES = (OF.samples as Record<string, string[]>)[String(OF_STEPS)];
const COPIED = SAMPLES.filter((s) => OF.names.includes(s)).length;
const UNIFORM = Math.log(27);
const f2 = (v: number) => v.toFixed(2);
const f3 = (v: number) => v.toFixed(3);
const oneIn = (loss: number) => { const n = Math.exp(loss); return n >= 1000 ? Math.round(+n.toPrecision(2)).toLocaleString('en-US') : n.toFixed(0); };
const mGap = (step: number) => mVal[step / MAIN.every] - mTrain[step / MAIN.every];

let sizes: { small: number; big: number } | undefined;
const weightCounts = () => (sizes ??= {
  small: new GPT(DEFAULT_CONFIG, rng(1)).paramCount(),
  big: new GPT({ ...DEFAULT_CONFIG, hidden: 64 }, rng(1)).paramCount(),
});

export const overfitting: Lesson = {
  id: 'overfitting',
  title: { en: 'Overfitting and Validation', zh: '過度擬合與驗證' },
  blurb: {
    en: 'A model that scores brilliantly on its practice names and fails on new ones. How to catch it, and what to do.',
    zh: '一個在練習用的名字上表現亮眼、遇到新名字就失靈的模型。怎麼抓出來，又該怎麼辦。',
  },
  minutes: 11,
  level: 'beginner',
  learn: [
    { text: { en: 'Why a low training loss can mislead', zh: '為什麼訓練損失低可能是假象' }, step: 0 },
    { text: { en: 'Training set vs validation set', zh: '訓練集與驗證集' }, step: 1 },
    { text: { en: 'What overfitting looks like in the numbers', zh: '過度擬合在數字上長什麼樣子' }, step: 2 },
    { text: { en: 'Memorizing instead of learning patterns', zh: '死背，而不是學到規律' }, step: 4 },
    { text: { en: 'Three remedies: more data, stop early, smaller model', zh: '三種對策：更多資料、提早停止、更小的模型' }, step: 6 },
  ],
  steps: [
    {
      short: { en: 'Too good', zh: '好得可疑' }, chapter: 'problem',
      title: { en: 'A loss that looks too good', zh: '好到可疑的損失' },
      body: [
        { en: `We train a model on just ${OF_NAMES} names for ${OF_STEPS.toLocaleString()} steps. Its training loss falls to ${f2(lastOf(ofTrain))}. The logged run (our ${M_STEPS.toLocaleString()}-step run on all ${SPLIT.train.toLocaleString()} training names) never gets below ${f2(Math.min(...mTrain))}.`, zh: `我們只用 ${OF_NAMES} 個名字訓練一個模型 ${OF_STEPS.toLocaleString()} 步。它的訓練損失降到 ${f2(lastOf(ofTrain))}。而紀錄的訓練（用全部 ${SPLIT.train.toLocaleString()} 個訓練名字跑了 ${M_STEPS.toLocaleString()} 步）損失從沒低於 ${f2(Math.min(...mTrain))}。` },
        { en: 'So is this a much better model? The training loss only measures the names it practised on. We have not yet asked it about any name it has not seen.', zh: '所以這是個好很多的模型嗎？訓練損失只量了它練習過的那些名字。我們還沒問過它任何沒看過的名字。' },
      ],
    },
    {
      short: { en: 'Hold out', zh: '先留下來' }, chapter: 'technique',
      title: { en: 'Keep some names back', zh: '留一些名字不給它看' },
      body: [
        { en: `Before training, we split our ${(SPLIT.train + SPLIT.val).toLocaleString()} names. Every fifth name (${SPLIT.val} of them) goes into the <b>validation set</b> and is never used for training. The other ${SPLIT.train.toLocaleString()} are the <b>training set</b>.`, zh: `訓練之前，先把 ${(SPLIT.train + SPLIT.val).toLocaleString()} 個名字分開。每五個抽一個（共 ${SPLIT.val} 個）放進<b>驗證集（validation set）</b>，完全不拿來訓練。其餘 ${SPLIT.train.toLocaleString()} 個是<b>訓練集（training set）</b>。` },
        { en: `The small experiment took its ${OF_NAMES} names from the training set. The loss on the validation names, the <b>validation loss</b>, shows how the model does on names it has never seen, which is what we actually care about. It is the “held-out loss” you have met in earlier lessons: the same ${SPLIT.val} names.`, zh: `這個小實驗的 ${OF_NAMES} 個名字是從訓練集裡挑的。模型在驗證集上的損失，也就是<b>驗證損失</b>，顯示它在從沒看過的名字上表現如何，這才是我們真正在乎的。前幾課看到的「驗證損失」就是它：同樣這 ${SPLIT.val} 個名字。` },
      ],
      callout: { kind: 'insight', text: { en: 'Like a teacher keeping the exam questions out of the homework: you only learn whether students understand if the test is new to them.', zh: '就像老師不會把考題放進作業裡：只有考的是新題目，才知道學生是不是真的懂。' } },
      changed: [{ sign: '+', text: { en: `Validation set: ${SPLIT.val} names`, zh: `驗證集：${SPLIT.val} 個名字` } }],
    },
    {
      short: { en: 'The gap', zh: '落差' }, chapter: 'technique',
      title: { en: 'Training loss down, validation loss up', zh: '訓練損失下降，驗證損失上升' },
      body: [
        { en: `Here is the same run with the validation loss added (red). For the first ${BEST_STEP} steps both fall (the validation loss only to ${f2(ofVal[bestIdx])}). Then they split: the training loss keeps falling while the validation loss climbs, all the way to ${f2(lastOf(ofVal))}.`, zh: `這是同一次訓練，加上了驗證損失（紅線）。前 ${BEST_STEP} 步兩者都下降（驗證損失只降到 ${f2(ofVal[bestIdx])}），接著就分道揚鑣：訓練損失繼續下降，驗證損失卻一路往上爬，最後到了 ${f2(lastOf(ofVal))}。` },
        { en: `That is far worse than random guessing (${f2(UNIFORM)}). This is <b>overfitting</b>: getting better on the training data while getting worse on new data.`, zh: `這比瞎猜（${f2(UNIFORM)}）還要糟得多。這就是<b>過度擬合（overfitting）</b>：在訓練資料上越來越好，在新資料上卻越來越差。` },
      ],
      changed: [{ sign: '+', text: { en: 'Validation loss', zh: '驗證損失' } }],
    },
    {
      short: { en: 'Confidently wrong', zh: '自信地錯' }, chapter: 'technique',
      title: { en: 'Worse than guessing means confidently wrong', zh: '比瞎猜還差，代表很有把握地猜錯' },
      body: [
        { en: `A loss of ${f2(lastOf(ofVal))} means that, on new names, the right next letter gets on average about 1 chance in ${oneIn(lastOf(ofVal))}. A model that guesses evenly would give it 1 in 27.`, zh: `損失 ${f2(lastOf(ofVal))} 的意思是：在新名字上，正確的下一個字母平均只得到大約 ${oneIn(lastOf(ofVal))} 分之 1 的機率。平均亂猜的模型都還有 27 分之 1。` },
        { en: 'The model is not merely unsure. It is very sure of the letters that came next in its own 30 names, and puts almost nothing on anything else.', zh: '這個模型不只是沒把握而已。它對自己那 30 個名字裡的下一個字母非常有把握，其他字母幾乎一點機率都不給。' },
      ],
      callout: { kind: 'note', text: { en: '“1 chance in N” is e raised to the loss. The Evaluation lesson calls this number perplexity.', zh: '「N 分之 1」中的 N 就是 e 的「損失」次方。「評估」那一課會把這個數字稱為困惑度。' } },
    },
    {
      short: { en: 'Memorized', zh: '死背' }, chapter: 'technique',
      title: { en: 'It writes back what it memorized', zh: '它寫出來的，都是背下來的' },
      body: [
        { en: `Ask the overfitted model to write names. All ${SAMPLES.length} come out below. ${COPIED} of ${SAMPLES.length} are copied exactly from its ${OF_NAMES} training names (highlighted in the list).`, zh: `請這個過度擬合的模型寫名字，${SAMPLES.length} 個結果都在下面。其中 ${COPIED} 個和它那 ${OF_NAMES} 個訓練名字一字不差（在名單中標了出來）。` },
        { en: 'It has not learned what names in general look like. It has learned these 30 names by heart. That is exactly what the low training loss was rewarding.', zh: '它沒有學到名字大致長什麼樣子，而是把這 30 個名字背了下來。低訓練損失獎勵的，正是這件事。' },
      ],
      changed: [{ sign: '+', text: { en: `${COPIED} of ${SAMPLES.length} samples copied`, zh: `${SAMPLES.length} 個中有 ${COPIED} 個是抄的` } }],
    },
    {
      short: { en: 'Healthy run', zh: '健康的訓練' }, chapter: 'solution',
      title: { en: 'A healthy run: the two lines stay close', zh: '健康的訓練：兩條線靠得很近' },
      body: [
        { en: `Now the logged run: the full ${SPLIT.train.toLocaleString()} training names, ${M_STEPS.toLocaleString()} steps. Training and validation loss fall together and stay close. At the end they are ${f2(lastOf(mTrain))} and ${f2(lastOf(mVal))}.`, zh: `現在看紀錄的訓練：完整的 ${SPLIT.train.toLocaleString()} 個訓練名字，${M_STEPS.toLocaleString()} 步。訓練損失和驗證損失一起下降，而且靠得很近。最後分別是 ${f2(lastOf(mTrain))} 和 ${f2(lastOf(mVal))}。` },
        { en: 'With thousands of names it cannot memorize them all, so the cheapest way to lower the loss is to learn patterns that also work on new names.', zh: '面對幾千個名字，它不可能全部背起來，所以降低損失最省力的方法，就是學會在新名字上也行得通的規律。' },
      ],
      changed: [{ sign: '~', text: { en: `${OF_NAMES} → ${SPLIT.train.toLocaleString()} training names`, zh: `訓練名字 ${OF_NAMES} → ${SPLIT.train.toLocaleString()} 個` } }],
    },
    {
      short: { en: 'Remedies', zh: '對策' }, chapter: 'technique',
      title: { en: 'Three ways to fight overfitting', zh: '對付過度擬合的三種方法' },
      body: [
        { en: '<b>More data</b>: harder to memorize, as we just saw. <b>Stop early</b>: watch the validation loss and keep the model from the moment it was lowest. In the small run that was only step ' + BEST_STEP + ', where the validation loss was still ' + f2(ofVal[bestIdx]) + ', barely better than guessing: stopping early limits the harm, but only more data gives a good model.', zh: '<b>更多資料</b>：越難背起來，剛剛已經看到了。<b>提早停止</b>：盯著驗證損失，保留它最低那一刻的模型。在那次小實驗裡，最低點只在第 ' + BEST_STEP + ' 步，那時驗證損失還有 ' + f2(ofVal[bestIdx]) + '，只比亂猜好一點：提早停止能減少傷害，但只有更多資料才能得到好模型。' },
        { en: '<b>Smaller model</b>: fewer weights means less room to store every example. The small experiment used a wider MLP, so it had more weights than the logged run’s model, for far fewer names.', zh: '<b>更小的模型</b>：權重越少，就越沒有空間把每個例子都存起來。那次小實驗用了更寬的 MLP，權重比紀錄的訓練所用的模型還多，名字卻少得多。' },
      ],
      callout: { kind: 'note', text: { en: 'We did not run a “smaller model” experiment here, so that third remedy is general advice, not something this page measured.', zh: '我們這裡沒有做「更小的模型」的實驗，所以第三種對策是一般原則，不是這一頁量出來的結果。' } },
      changed: [{ sign: '+', text: { en: 'More data · stop early · smaller model', zh: '更多資料 · 提早停止 · 更小的模型' } }],
    },
    {
      short: { en: 'Slow drift', zh: '慢慢漂開' }, chapter: 'problem',
      title: { en: 'Case: even the healthy run drifts apart', zh: '案例：連健康的訓練也會慢慢拉開' },
      body: [
        { en: `Look closely at the logged run. The gap between validation and training loss was ${f2(mGap(1000))} at step 1,000 and ${f2(mGap(M_STEPS))} at step ${M_STEPS.toLocaleString()}. Training loss keeps falling; validation loss has gone flat.`, zh: `仔細看紀錄的訓練。驗證損失和訓練損失的差距，在第 1,000 步是 ${f2(mGap(1000))}，到第 ${M_STEPS.toLocaleString()} 步變成 ${f2(mGap(M_STEPS))}。訓練損失還在下降，驗證損失卻已經持平。` },
        { en: 'This is a mild version of the same thing: late in training, part of the progress is memorizing. It is why we always check the validation loss, not the training loss, to decide when a model is good.', zh: '這是同一件事的輕微版本：訓練後期，有一部分進步其實是在死背。所以判斷模型好不好時，我們一定看驗證損失，而不是訓練損失。' },
      ],
      changed: [{ sign: '~', text: { en: `Gap ${f2(mGap(1000))} → ${f2(mGap(M_STEPS))}`, zh: `差距 ${f2(mGap(1000))} → ${f2(mGap(M_STEPS))}` } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Judge the model on names it has never seen', zh: '用沒看過的名字來評斷模型' },
      body: [
        { en: '<b>Training set</b>: what the model learns from.<br><b>Validation set</b>: held back, only used to check.<br><b>Overfitting</b>: training loss falls while validation loss rises; the model memorizes.<br><b>Remedies</b>: more data, stop early, a smaller model.', zh: '<b>訓練集</b>：模型拿來學習的資料。<br><b>驗證集</b>：預先保留，只用來檢查。<br><b>過度擬合</b>：訓練損失下降、驗證損失上升；模型在死背。<br><b>對策</b>：更多資料、提早停止、更小的模型。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Data and Tokenizer Pipeline. More data is the best remedy, so how is training data collected, cleaned and turned into tokens at scale?', zh: '下一課：資料與分詞流程。更多資料是最好的對策，那大規模的訓練資料是怎麼收集、清理，再變成詞元的？' } },
    },
  ],

  scene(step) {
    const items: Item[] = [];
    const overfitChart = step <= 4 || step === 6;
    if (overfitChart) {
      const series: { pts: number[]; tone: 'a' | 'bad' }[] = [{ pts: ofTrain, tone: 'a' }];
      if (step >= 2) series.push({ pts: ofVal, tone: 'bad' });
      items.push({
        k: 'ofc', t: 'chart', x: 90, y: 40, w: 760, h: step === 0 ? 300 : step === 4 ? 220 : 250, ymin: 0, ymax: 13, xmax: OF_STEPS, every: OF.every, series,
        xlabel: t({ en: `training steps (${OF_NAMES} names)`, zh: `訓練步數（${OF_NAMES} 個名字）` }), ylabel: t({ en: 'loss (lower is better)', zh: '損失（越低越好）' }),
        guide: { y: UNIFORM, label: t({ en: 'random guessing · 3.30', zh: '瞎猜 · 3.30' }) },
      });
    } else {
      items.push({
        k: 'mc', t: 'chart', x: 90, y: 40, w: 760, h: 250, ymin: 1.8, ymax: 3.6, xmax: M_STEPS, every: MAIN.every,
        series: [{ pts: mTrain, tone: 'a' }, { pts: mVal, tone: 'bad' }],
        xlabel: t({ en: `training steps (${SPLIT.train.toLocaleString()} names)`, zh: `訓練步數（${SPLIT.train.toLocaleString()} 個名字）` }), ylabel: t({ en: 'loss (lower is better)', zh: '損失（越低越好）' }),
        guide: { y: UNIFORM, label: t({ en: 'random guessing · 3.30', zh: '瞎猜 · 3.30' }) },
      });
    }

    if (step === 1) {
      const total = SPLIT.train + SPLIT.val, W = 760;
      const wT = (SPLIT.train / total) * W;
      items.push({ k: 'spT', t: 'box', x: 90, y: 360, w: wT - 6, h: 56, label: t({ en: `training set · ${SPLIT.train.toLocaleString()} names`, zh: `訓練集 · ${SPLIT.train.toLocaleString()} 個名字` }), sub: t({ en: 'the model learns from these', zh: '模型從這些學習' }), tone: 'a' });
      items.push({ k: 'spV', t: 'box', x: 90 + wT, y: 360, w: W - wT, h: 56, label: t({ en: `validation · ${SPLIT.val}`, zh: `驗證集 · ${SPLIT.val}` }), sub: t({ en: 'only for checking', zh: '只用來檢查' }), tone: 'b' });
      items.push({ k: 'spS', t: 'box', x: 90, y: 440, w: (OF_NAMES / total) * W + 4, h: 30, tone: 'a' });
      items.push({ k: 'spSt', t: 'text', x: 110, y: 461, size: 14, tone: 'muted', text: t({ en: `← the ${OF_NAMES} names of the small experiment (to scale)`, zh: `← 小實驗用的 ${OF_NAMES} 個名字（依比例）` }) });
    }

    if (step === 2 || step === 3) {
      const v = lastOf(ofVal), tr = lastOf(ofTrain);
      items.push({ k: 'ofT', t: 'text', x: 200, y: 380, anchor: 'middle', size: 15, tone: 'a', text: t({ en: 'training loss', zh: '訓練損失' }) });
      items.push({ k: 'ofTv', t: 'text', x: 200, y: 418, anchor: 'middle', size: 26, bold: true, mono: true, tone: 'a', text: f2(tr) });
      items.push({ k: 'ofV', t: 'text', x: 450, y: 380, anchor: 'middle', size: 15, tone: 'bad', text: t({ en: 'validation loss', zh: '驗證損失' }) });
      items.push({ k: 'ofVv', t: 'text', x: 450, y: 418, anchor: 'middle', size: 26, bold: true, mono: true, tone: 'bad', text: f2(v) });
      if (step === 3) {
        items.push({ k: 'oi', t: 'text', x: 700, y: 380, anchor: 'middle', size: 15, tone: 'bad', text: t({ en: 'right letter gets about', zh: '正確字母大約得到' }) });
        items.push({ k: 'oiv', t: 'text', x: 700, y: 418, anchor: 'middle', size: 24, bold: true, mono: true, tone: 'bad', text: t({ en: `1 in ${oneIn(v)}`, zh: `${oneIn(v)} 分之 1` }) });
        items.push({ k: 'oiu', t: 'text', x: 700, y: 450, anchor: 'middle', size: 13.5, tone: 'muted', text: t({ en: 'even guessing gives 1 in 27', zh: '瞎猜也有 27 分之 1' }) });
      }
    }

    if (step === 4) {
      items.push({ k: 'smL', t: 'text', x: 90, y: 318, size: 14, tone: 'muted', text: t({ en: 'the overfitted model writes:', zh: '過度擬合的模型寫出：' }) });
      SAMPLES.forEach((s, i) => items.push({ k: 'sm' + i, t: 'box', x: 90 + i * 96, y: 328, w: 88, h: 38, label: s, tone: OF.names.includes(s) ? 'b' : 'ink' }));
      items.push({ k: 'nlL', t: 'text', x: 90, y: 398, size: 14, tone: 'muted', text: t({ en: `its ${OF_NAMES} training names (copied ones in amber):`, zh: `它的 ${OF_NAMES} 個訓練名字（被抄的標成琥珀色）：` }) });
      OF.names.forEach((n, i) => {
        const hit = SAMPLES.includes(n);
        items.push({ k: 'nl' + i, t: 'text', x: 90 + (i % 10) * 76, y: 424 + Math.floor(i / 10) * 24, size: 14, mono: true, bold: hit, tone: hit ? 'b' : 'muted', text: n });
      });
    }

    if (step === 5 || step >= 7) {
      const g = step === 5 ? [M_STEPS] : step === 7 ? [1000, M_STEPS] : [];
      g.forEach((s, i) => {
        const x = 90 + (s / M_STEPS) * 760;
        items.push({ k: 'gp' + i, t: 'text', x: Math.min(x, 850), y: 330, anchor: s === M_STEPS ? 'end' : 'middle', size: 14, mono: true, tone: 'muted',
          text: t({ en: `step ${s.toLocaleString()}: gap ${f2(mGap(s))}`, zh: `第 ${s.toLocaleString()} 步：差距 ${f2(mGap(s))}` }) });
      });
      const rows = [
        [t({ en: `small run (${OF_NAMES} names)`, zh: `小實驗（${OF_NAMES} 個名字）` }), f2(lastOf(ofTrain)), f2(lastOf(ofVal)), 'bad'],
        [t({ en: `logged run (${SPLIT.train.toLocaleString()} names)`, zh: `紀錄的訓練（${SPLIT.train.toLocaleString()} 個名字）` }), f2(lastOf(mTrain)), f2(lastOf(mVal)), 'a'],
      ] as const;
      items.push({ k: 'rh1', t: 'text', x: 560, y: 384, anchor: 'middle', size: 14, bold: true, tone: 'muted', text: t({ en: 'training', zh: '訓練' }) });
      items.push({ k: 'rh2', t: 'text', x: 700, y: 384, anchor: 'middle', size: 14, bold: true, tone: 'muted', text: t({ en: 'validation', zh: '驗證' }) });
      rows.forEach((r, i) => {
        const y = 416 + i * 32;
        items.push({ k: 'rl' + i, t: 'text', x: 430, y, anchor: 'end', size: 14.5, text: r[0] });
        items.push({ k: 'rt' + i, t: 'text', x: 560, y, anchor: 'middle', size: 15, mono: true, text: r[1] });
        items.push({ k: 'rv' + i, t: 'text', x: 700, y, anchor: 'middle', size: 15, mono: true, bold: true, tone: r[3], text: r[2] });
      });
    }

    if (step === 6) {
      const W = weightCounts();
      const x = 90 + (BEST_STEP / OF_STEPS) * 760;
      items.push({ k: 'es', t: 'arrow', x1: x + 56, y1: 252, x2: x + 4, y2: 40 + 250 - (ofVal[bestIdx] / 13) * 250 + 2, tone: 'b' });
      items.push({ k: 'esT', t: 'text', x: x + 60, y: 262, size: 13.5, tone: 'b', text: t({ en: `stop here: step ${BEST_STEP}, validation ${f2(ofVal[bestIdx])}`, zh: `在這裡停：第 ${BEST_STEP} 步，驗證 ${f2(ofVal[bestIdx])}` }) });
      const cards = [
        { k: 'r1', label: t({ en: 'more data', zh: '更多資料' }), sub: t({ en: `${OF_NAMES} → ${SPLIT.train.toLocaleString()} names`, zh: `${OF_NAMES} → ${SPLIT.train.toLocaleString()} 個名字` }) },
        { k: 'r2', label: t({ en: 'stop early', zh: '提早停止' }), sub: t({ en: `best at step ${BEST_STEP}, not ${OF_STEPS.toLocaleString()}`, zh: `最佳在第 ${BEST_STEP} 步，不是 ${OF_STEPS.toLocaleString()}` }) },
        { k: 'r3', label: t({ en: 'smaller model', zh: '更小的模型' }), sub: t({ en: `${W.big.toLocaleString()} weights → ${W.small.toLocaleString()}`, zh: `${W.big.toLocaleString()} 個權重 → ${W.small.toLocaleString()}` }) },
      ];
      cards.forEach((c, i) => items.push({ k: c.k, t: 'box', x: 90 + i * 260, y: 370, w: 240, h: 64, label: c.label, sub: c.sub, tone: 'a' }));
      items.push({ k: 'rpn', t: 'text', x: 470, y: 470, anchor: 'middle', size: 13.5, tone: 'muted', text: t({ en: `small run: ${Math.round(W.big / OF_NAMES)} weights per name · logged run: ${(W.small / SPLIT.train).toFixed(1)} per name`, zh: `小實驗：每個名字 ${Math.round(W.big / OF_NAMES)} 個權重 · 紀錄的訓練：每個名字 ${(W.small / SPLIT.train).toFixed(1)} 個` }) });
    }

    const small = overfitChart;
    const metrics = [
      { label: { en: 'Training names', zh: '訓練名字數' }, value: small ? String(OF_NAMES) : SPLIT.train.toLocaleString() },
      { label: { en: 'Training loss', zh: '訓練損失' }, value: f3(small ? lastOf(ofTrain) : lastOf(mTrain)) },
      { label: { en: 'Validation loss', zh: '驗證損失' }, value: step === 0 ? '?' : f3(small ? lastOf(ofVal) : lastOf(mVal)) },
      ...(step === 4 ? [{ label: { en: 'Samples copied', zh: '抄來的樣本' }, value: `${COPIED} / ${SAMPLES.length}` }] : []),
    ];
    return {
      metrics,
      items,
      legend: [{ tone: 'a', text: { en: 'training loss', zh: '訓練損失' } }, ...(step >= 2 ? [{ tone: 'bad' as const, text: { en: 'validation loss', zh: '驗證損失' } }] : [])],
    };
  },
};
