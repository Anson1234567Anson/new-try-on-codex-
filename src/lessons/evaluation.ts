import type { Lesson } from './types';
import type { Item } from '../engine/scene';
import { t } from '../i18n';
import { GPT, type Weights } from '../model/gpt';
import { CHARS, VAL_NAMES, example } from '../model/data';
import { softmaxRow } from '../model/tensor';
import weights from '../model/weights.json';
import exp from '../model/experiments.json';

// The main 4,000-step run from experiments.json, with names written at seven checkpoints,
// plus a live check of the shipped model on held-out names.

const M = exp.main;
const EVERY = M.every;
const STEPS = M.steps;
const train = M.train.map((v) => v as number);
const val = M.val.map((v) => v as number);
const CK = Object.keys(M.samples).map(Number).sort((a, b) => a - b); // checkpoint steps with samples
const SAMPLES = M.samples as Record<string, string[]>;
const vAt = (s: number) => val[s / EVERY];
const tAt = (s: number) => train[s / EVERY];
const BEST_I = val.indexOf(Math.min(...val));
const BEST = BEST_I * EVERY;
// "flat": from here on the held-out loss stays within 0.02 of its best value
const FLAT = (val.reduce((last, v, i) => (v > val[BEST_I] + 0.02 ? i : last), 0) + 1) * EVERY;
const UNIFORM = 27;
const ppl = (loss: number) => Math.exp(loss);
const f1 = (v: number) => v.toFixed(1);
const f3 = (v: number) => v.toFixed(3);
const px = (s: number) => 90 + (s / STEPS) * 760;

let liveCache: ReturnType<typeof makeLive> | undefined;
function makeLive() {
  const model = GPT.fromJSON(weights as Weights);
  const all = VAL_NAMES.map(example);
  const loss = model.loss(all);
  const name = VAL_NAMES[0];
  const ex = example(name);
  const logits = model.forward(ex.x).logits;
  const probs = ex.y.map((y, i) => softmaxRow(logits.row(i))[y]);
  const nameLoss = -probs.reduce((s, p) => s + Math.log(p), 0) / probs.length;
  return { loss, name, ex, probs, nameLoss, count: VAL_NAMES.length };
}
const live = () => (liveCache ??= makeLive());

export const evaluation: Lesson = {
  id: 'evaluation',
  title: { en: 'Checkpoints and Evaluation', zh: '檢查點與評估' },
  blurb: {
    en: 'Save the model as it trains, measure each saved copy on held-out data, and keep the best one.',
    zh: '在訓練過程中把模型存起來，在保留的資料上評估每一份，留下最好的那一份。',
  },
  minutes: 12,
  level: 'beginner',
  learn: [
    { text: { en: 'What a checkpoint is and why runs save them', zh: '什麼是檢查點，以及為什麼訓練要存檢查點' }, step: 1 },
    { text: { en: 'Picking the best checkpoint by held-out loss', zh: '用保留資料上的損失挑出最好的檢查點' }, step: 4 },
    { text: { en: 'Perplexity: loss as “how many choices”', zh: '困惑度：把損失換成「幾選一」' }, step: 5 },
    { text: { en: 'Reading samples to compare checkpoints by eye', zh: '讀樣本，用眼睛比較檢查點' }, step: 7 },
    { text: { en: 'Why real evaluation also uses benchmarks and people', zh: '為什麼真實的評估還要用基準測驗和人工判斷' }, step: 8 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'Which model do we keep?', zh: '要留下哪一個模型？' },
      body: [
        { en: `This is our main training run: ${STEPS.toLocaleString()} steps with Adam, warmup and cosine decay. The training loss falls the whole way, from ${f3(train[0])} to ${f3(train[train.length - 1])}.`, zh: `這是我們的主要訓練：用 Adam、暖身與餘弦衰減，共 ${STEPS.toLocaleString()} 步。訓練損失一路下降，從 ${f3(train[0])} 降到 ${f3(train[train.length - 1])}。` },
        { en: 'So is the model at the very end the one to keep? The training loss cannot tell us: the last lesson showed it keeps improving even when the model is only memorizing.', zh: '那麼，最後那一刻的模型就是該留下的嗎？訓練損失回答不了：上一課看到，就算模型只是在死背，訓練損失也會一直進步。' },
      ],
    },
    {
      short: { en: 'Checkpoints', zh: '檢查點' }, chapter: 'technique',
      title: { en: 'Save copies along the way', zh: '一路存下副本' },
      body: [
        { en: `A <b>checkpoint</b> is a saved copy of all the model’s weights (here ${M.params.toLocaleString()}) at one moment of training, usually with the optimizer’s running averages too. This run was measured every ${EVERY} steps; the chips mark ${CK.length} of those moments, where we also had it write names.`, zh: `<b>檢查點（checkpoint）</b>是把模型在訓練某一刻的全部權重（這裡有 ${M.params.toLocaleString()} 個）存成一份副本，通常連最佳化器的移動平均也一起存。這次訓練每 ${EVERY} 步量測一次；下面的小方塊標出其中 ${CK.length} 個時間點，那幾次我們也請它寫了名字。` },
        { en: 'Checkpoints let a long run restart after a crash instead of from zero, and they let us compare the model at different moments after the fact.', zh: '有了檢查點，長時間的訓練萬一當掉，可以從存檔處接著跑，而不用從頭來；事後也能比較模型在不同時間點的表現。' },
      ],
      changed: [{ sign: '+', text: { en: `${CK.length} checkpoints`, zh: `${CK.length} 個檢查點` } }],
    },
    {
      short: { en: 'Evaluate', zh: '評估' }, chapter: 'technique',
      title: { en: 'Measure each checkpoint on held-out names', zh: '用保留的名字評估每個檢查點' },
      body: [
        { en: `To <b>evaluate</b> a checkpoint, run it on the ${exp.split.val} validation names it never trained on and measure the loss. No weights change; we only measure.`, zh: `要<b>評估（evaluate）</b>一個檢查點，就讓它跑過那 ${exp.split.val} 個從沒訓練過的驗證名字，量出損失。權重完全不動，只是量測。` },
        { en: 'The red line is that held-out loss, measured every 50 steps. Each chip now shows its checkpoint’s value.', zh: '紅線就是保留資料上的損失，每 50 步量一次。每個小方塊上現在也標出了該檢查點的數值。' },
      ],
      changed: [{ sign: '+', text: { en: 'Held-out loss for every checkpoint', zh: '每個檢查點在保留資料上的損失' } }],
    },
    {
      short: { en: 'Flat', zh: '持平' }, chapter: 'technique',
      title: { en: 'The held-out loss goes flat', zh: '保留資料上的損失變平了' },
      body: [
        { en: `From about step ${FLAT.toLocaleString()} on, the held-out loss stays within 0.02 of its best value. It went from ${f3(vAt(FLAT))} to ${f3(val[val.length - 1])} over the rest of the run.`, zh: `大約從第 ${FLAT.toLocaleString()} 步開始，保留資料上的損失就一直停在最佳值的 0.02 以內。之後到結束，它從 ${f3(vAt(FLAT))} 變成 ${f3(val[val.length - 1])}。` },
        { en: `Meanwhile the training loss kept falling, from ${f3(tAt(FLAT))} to ${f3(train[train.length - 1])}. Those last ${(STEPS - FLAT).toLocaleString()} steps mostly taught it the training names better, not names in general.`, zh: `同一段時間，訓練損失卻繼續從 ${f3(tAt(FLAT))} 降到 ${f3(train[train.length - 1])}。最後這 ${(STEPS - FLAT).toLocaleString()} 步，主要是讓它把訓練名字記得更熟，而不是更懂一般的名字。` },
      ],
      changed: [{ sign: '+', text: { en: `Flat from step ${FLAT.toLocaleString()}`, zh: `第 ${FLAT.toLocaleString()} 步之後持平` } }],
    },
    {
      short: { en: 'Pick the best', zh: '挑最好的' }, chapter: 'solution',
      title: { en: 'Keep the checkpoint with the lowest held-out loss', zh: '留下保留資料損失最低的檢查點' },
      body: [
        { en: `Across all ${val.length} measurements (one every ${EVERY} steps, each a checkpoint we could keep), the lowest held-out loss is ${f3(val[BEST_I])} at step ${BEST.toLocaleString()}. That is the checkpoint to keep.`, zh: `在全部 ${val.length} 次量測中（每 ${EVERY} 步一次，每次都是一個可留下的檢查點），保留資料上損失最低的是第 ${BEST.toLocaleString()} 步的 ${f3(val[BEST_I])}。該留下的就是那個檢查點。` },
        { en: `Honestly, on this run the difference is small: the final model scores ${f3(val[val.length - 1])}. Picking the best checkpoint matters much more when a run overfits badly, like the 30-name run in the last lesson.`, zh: `老實說，這次訓練差別不大：最後的模型是 ${f3(val[val.length - 1])}。當訓練嚴重過度擬合時（像上一課只用 30 個名字那次），挑對檢查點就重要得多。` },
      ],
      changed: [{ sign: '+', text: { en: `Best: step ${BEST.toLocaleString()}`, zh: `最佳：第 ${BEST.toLocaleString()} 步` } }],
    },
    {
      short: { en: 'Perplexity', zh: '困惑度' }, chapter: 'technique',
      title: { en: 'Perplexity: how many letters is it choosing between?', zh: '困惑度：它在幾個字母之間猶豫？' },
      body: [
        { en: `A loss like ${f3(val[BEST_I])} is hard to picture. <b>Perplexity</b> turns it into something you can: e raised to the loss. e<sup>${f3(val[BEST_I])}</sup> ≈ ${f1(ppl(val[BEST_I]))}.`, zh: `像 ${f3(val[BEST_I])} 這樣的損失很難想像。<b>困惑度（perplexity）</b>把它換成好懂的數字：e 的「損失」次方。e<sup>${f3(val[BEST_I])}</sup> ≈ ${f1(ppl(val[BEST_I]))}。` },
        { en: `It means: on average the model is as unsure as if it were choosing evenly among about ${Math.round(ppl(val[BEST_I]))} letters. The untrained model scored ${f1(ppl(val[0]))}, worse than picking evenly among all ${UNIFORM}.`, zh: `意思是：平均來說，模型的猶豫程度，就像在大約 ${Math.round(ppl(val[BEST_I]))} 個字母裡平均挑一個。未訓練的模型是 ${f1(ppl(val[0]))}，比在全部 ${UNIFORM} 個裡平均亂挑還差。` },
      ],
      callout: { kind: 'note', text: { en: 'Lower is better. A perfect model that always knew the next letter would have perplexity 1.', zh: '越低越好。一個永遠知道下一個字母的完美模型，困惑度是 1。' } },
      changed: [{ sign: '+', text: { en: `Perplexity ${f1(ppl(val[0]))} → ${f1(ppl(val[BEST_I]))}`, zh: `困惑度 ${f1(ppl(val[0]))} → ${f1(ppl(val[BEST_I]))}` } }],
    },
    {
      short: { en: 'Letter by letter', zh: '逐字母' }, chapter: 'technique',
      title: { en: 'Perplexity, letter by letter', zh: '逐字母看困惑度' },
      body: [
        { en: `Here is the model that ships with this course, run live on the held-out name “${live().name}”. Under each letter: the probability it gave the right answer, written as “1 in N”.`, zh: `這是本課程附帶的模型，現場跑在保留名字「${live().name}」上。每個字母下方是它給正確答案的機率，寫成「N 選 1」。` },
        { en: 'Some letters are easy (1 in 2 or 3), some are a real surprise (1 in 20 or more). The perplexity is a kind of average of these N values (the geometric mean), over every letter of every held-out name.', zh: '有些字母很好猜（2、3 選 1），有些則出乎意料（20 選 1 以上）。困惑度就是把所有保留名字中每個字母的這些 N 做一種平均（幾何平均）。' },
      ],
      changed: [{ sign: '+', text: { en: 'Live check on a held-out name', zh: '在保留名字上現場檢查' } }],
    },
    {
      short: { en: 'Read samples', zh: '讀樣本' }, chapter: 'technique',
      title: { en: 'Read what each checkpoint writes', zh: '讀讀每個檢查點寫了什麼' },
      body: [
        { en: 'Numbers are not everything. Each checkpoint wrote 8 names (same random seed every time). Reading them is a quick sanity check that the numbers mean what we think.', zh: '數字不是全部。每個檢查點都寫了 8 個名字（每次用同一個隨機種子）。讀一讀，可以快速確認數字代表的意義跟我們想的一樣。' },
        { en: `Step 0 is noise. By step 250 the names have vowels in sensible places. After step 1,000 the changes are hard to judge by eye, matching the flat held-out loss.`, zh: '第 0 步是雜訊。到第 250 步，母音已經出現在合理的位置。第 1,000 步之後的差別，用眼睛已經很難判斷，正好和保留資料上持平的損失一致。' },
      ],
      changed: [{ sign: '+', text: { en: `Samples from ${CK.length} checkpoints`, zh: `${CK.length} 個檢查點的樣本` } }],
    },
    {
      short: { en: 'Beyond loss', zh: '損失之外' }, chapter: 'problem',
      title: { en: 'Case: a good loss is not the whole story', zh: '案例：損失好看，不代表一切' },
      body: [
        { en: 'For a model that writes names, held-out loss and a quick read are enough. For a large model that answers questions, writes code or chats, a low loss does not tell you whether its answers are correct, useful or safe.', zh: '對一個寫名字的模型來說，保留資料上的損失加上讀幾個樣本就夠了。但對一個會回答問題、寫程式、聊天的大型模型，低損失並不能告訴你它的回答是否正確、有用或安全。' },
        { en: 'So real evaluations add <b>benchmarks</b> (fixed sets of tasks with answers that can be scored automatically) and <b>human judgment</b> (people comparing and rating outputs).', zh: '所以真實的評估還會加上<b>基準測驗（benchmark）</b>：一組固定的題目，答案可以自動評分；以及<b>人工判斷</b>：由人來比較並評分模型的輸出。' },
      ],
      callout: { kind: 'note', text: { en: 'Benchmarks can leak into training data, and then a model scores well by memorizing them. It is the overfitting problem again, one level up.', zh: '基準測驗的題目可能混進訓練資料，模型就能靠背答案拿高分。這又是過度擬合的問題，只是換到更高的層次。' } },
      changed: [{ sign: '+', text: { en: 'Benchmarks · human judgment', zh: '基準測驗 · 人工判斷' } }],
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Save, measure, choose', zh: '存檔、量測、挑選' },
      body: [
        { en: '<b>Checkpoint</b>: a saved copy of the weights at one moment.<br><b>Evaluate</b>: measure each one on held-out data, without training on it.<br><b>Pick</b>: keep the checkpoint with the best held-out score.<br><b>Perplexity</b>: e<sup>loss</sup>, “choosing among about N letters”.<br><b>Beyond loss</b>: samples, benchmarks and human judgment.', zh: '<b>檢查點</b>：某一刻權重的存檔副本。<br><b>評估</b>：在保留資料上量測每一份，不拿它訓練。<br><b>挑選</b>：留下保留資料上分數最好的檢查點。<br><b>困惑度</b>：e<sup>損失</sup>，「大約在 N 個字母中挑一個」。<br><b>損失之外</b>：樣本、基準測驗與人工判斷。' },
      ],
      callout: { kind: 'next', text: { en: 'Next: Pretraining vs Fine-Tuning. Start from a good checkpoint and teach it a new, narrower task.', zh: '下一課：預訓練與微調。從一個好的檢查點出發，教它一個更專門的新任務。' } },
    },
  ],

  scene(step) {
    const items: Item[] = [];
    const chartOn = step <= 5 || step >= 8;
    if (chartOn) {
      const series: { pts: number[]; tone: 'a' | 'bad' }[] = [{ pts: train, tone: 'a' }];
      if (step >= 2) series.push({ pts: val, tone: 'bad' });
      items.push({
        k: 'ch', t: 'chart', x: 90, y: 40, w: 760, h: step === 8 ? 190 : step === 9 ? 300 : 250, ymin: 1.8, ymax: 3.6, xmax: STEPS, every: EVERY, series,
        xlabel: t({ en: 'training steps', zh: '訓練步數' }), ylabel: t({ en: 'loss (lower is better)', zh: '損失（越低越好）' }),
      });
    }

    if (step >= 1 && step <= 4) {
      CK.forEach((s, i) => {
        const x = 90 + i * 110;
        items.push({ k: 'ck' + i, t: 'box', x, y: 390, w: 100, h: step >= 2 ? 54 : 40, label: t({ en: `step ${s.toLocaleString()}`, zh: `第 ${s.toLocaleString()} 步` }),
          sub: step >= 2 ? f3(vAt(s)) : undefined, tone: 'ink' });
        items.push({ k: 'cka' + i, t: 'arrow', x1: x + 50, y1: 384, x2: px(s), y2: 300, dashed: true });
      });
      items.push({ k: 'ckL', t: 'text', x: 90, y: 466, size: 13.5, tone: 'muted', text: step >= 2
        ? t({ en: 'each saved copy, with its held-out loss', zh: '每一份存檔，以及它在保留資料上的損失' })
        : t({ en: `each chip: a saved copy of all ${M.params.toLocaleString()} weights`, zh: `每個方塊：全部 ${M.params.toLocaleString()} 個權重的存檔` }) });
    }
    if (step === 3) {
      items.push({ k: 'flat', t: 'box', x: px(FLAT), y: 160, w: px(STEPS) - px(FLAT), h: 70, tone: 'b' });
      items.push({ k: 'flatT', t: 'text', x: px(FLAT) + 8, y: 152, size: 13.5, tone: 'b', text: t({ en: `flat from step ${FLAT.toLocaleString()}`, zh: `第 ${FLAT.toLocaleString()} 步起持平` }) });
    }
    if (step === 4 || step === 9) {
      const h = step === 9 ? 300 : 250;
      const y = 40 + h - ((val[BEST_I] - 1.8) / 1.8) * h;
      items.push({ k: 'best', t: 'arrow', x1: px(BEST), y1: y - 60, x2: px(BEST), y2: y - 6, tone: 'a' });
      items.push({ k: 'bestT', t: 'text', x: px(BEST), y: y - 68, anchor: 'middle', size: 14, bold: true, tone: 'a', text: t({ en: `best: step ${BEST.toLocaleString()} · ${f3(val[BEST_I])}`, zh: `最佳：第 ${BEST.toLocaleString()} 步 · ${f3(val[BEST_I])}` }) });
    }

    if (step === 9) {
      items.push({ k: 'rcp', t: 'text', x: 470, y: 420, anchor: 'middle', size: 17, bold: true, mono: true, tone: 'a', text: t({ en: `keep step ${BEST.toLocaleString()}: loss ${f3(val[BEST_I])} → perplexity ${f1(ppl(val[BEST_I]))}`, zh: `留下第 ${BEST.toLocaleString()} 步：損失 ${f3(val[BEST_I])} → 困惑度 ${f1(ppl(val[BEST_I]))}` }) });
      items.push({ k: 'rcp2', t: 'text', x: 470, y: 455, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: `about as unsure as choosing among ${Math.round(ppl(val[BEST_I]))} letters`, zh: `猶豫程度大約等於在 ${Math.round(ppl(val[BEST_I]))} 個字母裡挑一個` }) });
    }

    if (step === 5) {
      const cards = [
        { k: 'p0', label: t({ en: 'untrained (step 0)', zh: '未訓練（第 0 步）' }), loss: val[0], tone: 'bad' as const },
        { k: 'pu', label: t({ en: 'guessing evenly', zh: '平均亂猜' }), loss: Math.log(UNIFORM), tone: 'ink' as const },
        { k: 'pb', label: t({ en: `best (step ${BEST.toLocaleString()})`, zh: `最佳（第 ${BEST.toLocaleString()} 步）` }), loss: val[BEST_I], tone: 'a' as const },
      ];
      cards.forEach((c, i) => {
        const x = 90 + i * 260;
        items.push({ k: c.k, t: 'box', x, y: 350, w: 240, h: 96, label: c.label, tone: c.tone });
        items.push({ k: c.k + 'v', t: 'text', x: x + 120, y: 428, anchor: 'middle', size: 15, mono: true, text: t({ en: `loss ${f3(c.loss)} → ${f1(ppl(c.loss))}`, zh: `損失 ${f3(c.loss)} → ${f1(ppl(c.loss))}` }) });
      });
      items.push({ k: 'pT', t: 'text', x: 470, y: 495, anchor: 'middle', size: 13.5, tone: 'muted', text: t({ en: 'perplexity = e^loss: “as unsure as choosing evenly among this many letters”', zh: '困惑度 = e^損失：「猶豫程度相當於在這麼多個字母中平均挑一個」' }) });
    }

    if (step === 6) {
      const L = live();
      items.push({ k: 'lvL', t: 'text', x: 450, y: 70, anchor: 'middle', size: 14.5, tone: 'muted', text: t({ en: `each letter of “${L.name}” as the model must predict it, from the letters before it`, zh: `模型要依據前面的字母，逐一預測「${L.name}」的每個字母` }) });
      L.ex.y.forEach((c, i) => {
        const x = 80 + i * 96;
        const p = L.probs[i];
        items.push({ k: 'lt' + i, t: 'token', x, y: 130, text: CHARS[c], tone: c === 0 ? 'b' : 'a' });
        items.push({ k: 'lp' + i, t: 'text', x: x + 22, y: 210, anchor: 'middle', size: 14, mono: true, text: Math.round(p * 100) + '%' });
        items.push({ k: 'ln' + i, t: 'text', x: x + 22, y: 240, anchor: 'middle', size: 15, bold: true, tone: 1 / p > 10 ? 'bad' : 'a', text: t({ en: `1 in ${f1(1 / p)}`, zh: `${f1(1 / p)} 選 1` }) });
      });
      items.push({ k: 'lvN', t: 'text', x: 450, y: 320, anchor: 'middle', size: 15, mono: true, text: t({ en: `this name: loss ${f3(L.nameLoss)} → perplexity ${f1(ppl(L.nameLoss))}`, zh: `這個名字：損失 ${f3(L.nameLoss)} → 困惑度 ${f1(ppl(L.nameLoss))}` }) });
      items.push({ k: 'lvAll', t: 'text', x: 450, y: 370, anchor: 'middle', size: 17, bold: true, mono: true, tone: 'a', text: t({ en: `all ${L.count} held-out names: loss ${f3(L.loss)} → perplexity ${f1(ppl(L.loss))}`, zh: `全部 ${L.count} 個保留名字：損失 ${f3(L.loss)} → 困惑度 ${f1(ppl(L.loss))}` }) });
      items.push({ k: 'lvE', t: 'text', x: 450, y: 420, anchor: 'middle', size: 13.5, tone: 'muted', text: t({ en: 'the last chip, “.”, is the end of the name: the model must also predict when to stop', zh: '最後的「.」代表名字結束：模型也得預測什麼時候該停' }) });
    }

    if (step === 7) {
      const cols = [100, 210, 320];
      items.push({ k: 'sh0', t: 'text', x: cols[0], y: 50, anchor: 'end', size: 13.5, bold: true, tone: 'muted', text: t({ en: 'step', zh: '步數' }) });
      items.push({ k: 'sh1', t: 'text', x: cols[1], y: 50, anchor: 'end', size: 13.5, bold: true, tone: 'muted', text: t({ en: 'perplexity', zh: '困惑度' }) });
      items.push({ k: 'sh2', t: 'text', x: cols[2] - 60, y: 50, size: 13.5, bold: true, tone: 'muted', text: t({ en: 'names it writes', zh: '寫出的名字' }) });
      CK.forEach((s, i) => {
        const y = 92 + i * 58;
        items.push({ k: 'ck' + i, t: 'box', x: 30, y: y - 26, w: 840, h: 44, tone: s === 0 ? 'bad' : 'ink' });
        items.push({ k: 'ss' + i, t: 'text', x: cols[0], y: y + 2, anchor: 'end', size: 15, mono: true, bold: true, text: s.toLocaleString() });
        items.push({ k: 'sp' + i, t: 'text', x: cols[1], y: y + 2, anchor: 'end', size: 15, mono: true, tone: 'a', text: f1(ppl(vAt(s))) });
        items.push({ k: 'sn' + i, t: 'text', x: cols[2] - 60, y: y + 2, size: 15, mono: true, text: SAMPLES[String(s)].join('  ') });
      });
    }

    if (step === 8) {
      const cards = [
        { k: 'e1', label: t({ en: 'held-out loss', zh: '保留資料損失' }), sub: t({ en: 'used in this lesson', zh: '本課用過' }), tone: 'a' as const },
        { k: 'e2', label: t({ en: 'reading samples', zh: '讀樣本' }), sub: t({ en: 'used in this lesson', zh: '本課用過' }), tone: 'a' as const },
        { k: 'e3', label: t({ en: 'benchmarks', zh: '基準測驗' }), sub: t({ en: 'fixed tasks, scored answers', zh: '固定題目，自動評分' }), tone: 'b' as const },
        { k: 'e4', label: t({ en: 'human judgment', zh: '人工判斷' }), sub: t({ en: 'people compare outputs', zh: '由人比較輸出' }), tone: 'b' as const },
      ];
      cards.forEach((c, i) => items.push({ k: c.k, t: 'box', x: 50 + i * 205, y: 330, w: 190, h: 70, label: c.label, sub: c.sub, tone: c.tone }));
      items.push({ k: 'e5', t: 'text', x: 450, y: 450, anchor: 'middle', size: 14, tone: 'muted', text: t({ en: 'teal: enough for our name model · amber: what large models need as well', zh: '青色：我們的名字模型這樣就夠 · 琥珀色：大型模型還需要的' }) });
    }

    const metrics = step === 6
      ? [
        { label: { en: 'Held-out names', zh: '保留名字數' }, value: String(live().count) },
        { label: { en: 'Held-out loss', zh: '保留資料損失' }, value: f3(live().loss) },
        { label: { en: 'Perplexity', zh: '困惑度' }, value: f1(ppl(live().loss)) },
      ]
      : [
        { label: { en: 'Checkpoints', zh: '檢查點' }, value: step === 0 ? '—' : String(CK.length) },
        { label: { en: 'Final train loss', zh: '最後訓練損失' }, value: f3(train[train.length - 1]) },
        { label: { en: 'Best held-out', zh: '最佳保留損失' }, value: step >= 4 ? `${f3(val[BEST_I])} @${BEST}` : step >= 2 ? f3(val[BEST_I]) : '?' },
        { label: { en: 'Perplexity', zh: '困惑度' }, value: step >= 5 ? f1(ppl(val[BEST_I])) : '—' },
      ];
    return {
      metrics,
      items,
      legend: chartOn ? [{ tone: 'a', text: { en: 'training loss', zh: '訓練損失' } }, ...(step >= 2 ? [{ tone: 'bad' as const, text: { en: 'held-out loss', zh: '保留資料損失' } }] : [])] : undefined,
    };
  },
};
