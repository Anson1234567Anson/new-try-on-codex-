import type { Lesson } from './types';
import type { Item, Tone } from '../engine/scene';
import { t } from '../i18n';
import { TRAIN_NAMES, VAL_NAMES } from '../model/data';
import experiments from '../model/experiments.json';

// Real runs from scripts/experiments.ts: the shipped (pretrained) model fine-tuned on 40 names
// that end in "a", compared with a new model trained only on those 40 names.

const FT = experiments.finetune;
const FT_NAMES = TRAIN_NAMES.filter((n) => n.endsWith('a')).slice(0, FT.trainNames); // the same 40 the run used
const VAL_A = VAL_NAMES.filter((n) => n.endsWith('a')).length;
const STEPS = (FT.full.val.length - 1) * FT.full.every;

const best = (v: number[]) => { const m = Math.min(...v); return { loss: m, step: v.indexOf(m) * FT.full.every }; };
const BEST = { full: best(FT.full.val), scratch: best(FT.scratch.val) };
const pct = (v: number) => (v * 100).toFixed(1) + '%';
const fmt = (n: number) => n.toLocaleString('en-US');
const last = (a: number[]) => a[a.length - 1];
const SAMPLES = { base: FT.baseSamples, full: FT.full.samples['80'], scratch: FT.scratch.samples['80'] };
const copied = SAMPLES.full.filter((n) => FT_NAMES.includes(n));

export const fineTuning: Lesson = {
  id: 'fine-tuning',
  title: { en: 'Pretraining vs Fine-Tuning', zh: '預訓練與微調' },
  blurb: {
    en: 'Teach a trained model a new habit with just 40 examples, and see why starting from scratch does worse.',
    zh: '只用 40 個範例，教一個訓練好的模型新習慣，並看看為什麼從頭訓練反而比較差。',
  },
  minutes: 11,
  level: 'intermediate',
  learn: [
    { text: { en: 'What pretraining gives a model', zh: '預訓練帶給模型什麼' }, step: 1 },
    { text: { en: 'How fine-tuning adapts it with little data', zh: '微調如何用少量資料調整模型' }, step: 3 },
    { text: { en: 'Why fine-tuning beats training from scratch', zh: '為什麼微調勝過從頭訓練' }, step: 4 },
    { text: { en: 'Why fine-tuning must stop early', zh: '為什麼微調要及早停止' }, step: 5 },
    { text: { en: 'How chat assistants are made', zh: '聊天助理是怎麼做出來的' }, step: 7 },
  ],
  steps: [
    {
      short: { en: 'The problem', zh: '問題' }, chapter: 'problem',
      title: { en: 'We want a different style of name', zh: '我們想要另一種風格的名字' },
      body: [
        { en: `Our trained model writes names like these. Only <b>${pct(FT.baseEndsA)}</b> of 400 names it wrote end in “a”, close to the ${pct(FT.allNamesEndsA)} in our whole name list.`, zh: `我們訓練好的模型會寫出這樣的名字。它寫的 400 個名字裡，只有 <b>${pct(FT.baseEndsA)}</b> 以「a」結尾，和整份名字清單的 ${pct(FT.allNamesEndsA)} 差不多。` },
        { en: 'Suppose we want names that end in “a”, like maria or tessa. We have only a few dozen examples. Do we have to start all over?', zh: '假設我們想要以「a」結尾的名字，例如 maria 或 tessa。可是我們手上只有幾十個範例。難道要全部從頭來過嗎？' },
      ],
      callout: { kind: 'insight', text: { en: 'This is the everyday situation with large models: one expensive general model, many specific jobs with little data each.', zh: '這正是大型模型的日常處境：一個昂貴的通用模型，加上許多各自只有少量資料的特定任務。' } },
    },
    {
      short: { en: 'Pretraining', zh: '預訓練' }, chapter: 'technique',
      title: { en: 'Pretraining: learn the general skill', zh: '預訓練：學會通用的能力' },
      body: [
        { en: `What we have done so far is <b>pretraining</b>: training on all ${fmt(TRAIN_NAMES.length)} training names, 6,000 steps, to predict the next letter.`, zh: `我們到目前為止做的，就是<b>預訓練</b>：用全部 ${fmt(TRAIN_NAMES.length)} 個訓練名字，訓練 6,000 步，學習預測下一個字母。` },
        { en: 'The model learned how names work in general: which letters follow which, how long names are, how they end. That knowledge is useful for any kind of name.', zh: '模型因此學到名字的一般規律：哪些字母常接在哪些字母後面、名字大概多長、怎麼結尾。這些知識對任何一種名字都有用。' },
      ],
      changed: [{ sign: '+', text: { en: 'Pretrained model: all training names', zh: '預訓練模型：全部訓練名字' } }],
    },
    {
      short: { en: 'Small data', zh: '少量資料' }, chapter: 'technique',
      title: { en: 'Fine-tuning: keep training on a small set', zh: '微調：在一小份資料上繼續訓練' },
      body: [
        { en: `<b>Fine-tuning</b> means: start from the pretrained weights and keep training, but only on the new examples. Ours are these ${FT.trainNames} training names that end in “a”.`, zh: `<b>微調</b>的意思是：從預訓練好的權重出發，繼續訓練，但只用新的範例。我們的範例就是這 ${FT.trainNames} 個以「a」結尾的訓練名字。` },
        { en: `We use a small learning rate (0.003) so the model adjusts gently instead of forgetting what it knows. To check it fairly we use the ${VAL_A} held-out names ending in “a”.`, zh: `我們用很小的學習率（0.003），讓模型溫和地調整，而不是忘掉它已經會的東西。為了公平檢查，我們用保留集中 ${VAL_A} 個以「a」結尾的名字。` },
      ],
      changed: [{ sign: '+', text: { en: `${FT.trainNames} fine-tuning names`, zh: `${FT.trainNames} 個微調用的名字` } }],
    },
    {
      short: { en: 'Fine-tune', zh: '微調' }, chapter: 'solution',
      title: { en: 'A few steps change the habit', zh: '短短幾步就改變了習慣' },
      body: [
        { en: `The held-out loss on names ending in “a” drops from ${FT.baseValLoss.toFixed(2)} to <b>${BEST.full.loss.toFixed(2)}</b> after just ${BEST.full.step} steps.`, zh: `在以「a」結尾的保留名字上，損失只花了 ${BEST.full.step} 步，就從 ${FT.baseValLoss.toFixed(2)} 降到 <b>${BEST.full.loss.toFixed(2)}</b>。` },
        { en: `After the ${STEPS} steps, <b>${pct(FT.full.endsA)}</b> of the names it writes end in “a”, up from ${pct(FT.baseEndsA)}.`, zh: `跑完 ${STEPS} 步後，它寫的名字有 <b>${pct(FT.full.endsA)}</b> 以「a」結尾，原本只有 ${pct(FT.baseEndsA)}。` },
      ],
      changed: [{ sign: '~', text: { en: `Ends in “a”: ${pct(FT.baseEndsA)} → ${pct(FT.full.endsA)}`, zh: `以「a」結尾：${pct(FT.baseEndsA)} → ${pct(FT.full.endsA)}` } }],
    },
    {
      short: { en: 'vs scratch', zh: '對照從頭訓練' }, chapter: 'technique',
      title: { en: 'Why not just train on the 40 names?', zh: '為什麼不直接用這 40 個名字訓練？' },
      body: [
        { en: `For comparison, a brand-new model trained only on the same ${FT.trainNames} names (amber) starts from guessing and its best held-out loss is <b>${BEST.scratch.loss.toFixed(2)}</b>, clearly worse than ${BEST.full.loss.toFixed(2)}.`, zh: `做個對照：一個全新的模型只用同樣這 ${FT.trainNames} 個名字訓練（琥珀色），它從亂猜開始，在保留集上最好的損失是 <b>${BEST.scratch.loss.toFixed(2)}</b>，明顯比 ${BEST.full.loss.toFixed(2)} 差。` },
        { en: `It does learn the ending (${pct(FT.scratch.endsA)} end in “a”), but 40 names cannot teach it everything else about names. The fine-tuned model brings that knowledge from pretraining.`, zh: `它確實學會了結尾（${pct(FT.scratch.endsA)} 以「a」結尾），但 40 個名字教不了它名字的其他規律。微調的模型則把這些知識從預訓練帶了過來。` },
      ],
      callout: { kind: 'insight', text: { en: 'Pretraining does the expensive general learning once. Fine-tuning reuses it, so a small dataset is enough.', zh: '預訓練一次做完昂貴的通用學習；微調重複利用這些成果，所以少量資料就夠了。' } },
      changed: [{ sign: '+', text: { en: 'From-scratch model for comparison', zh: '加入從頭訓練的模型做對照' } }],
    },
    {
      short: { en: 'Case: too long', zh: '案例：練太久' }, chapter: 'problem',
      title: { en: 'Case: fine-tune too long and it memorizes', zh: '案例：微調太久就會死背' },
      body: [
        { en: `Keep going and the training loss keeps falling (to ${last(FT.full.train).toFixed(2)}), but the held-out loss turns around after step ${BEST.full.step} and climbs to ${last(FT.full.val).toFixed(2)}. That is <b>overfitting</b>: 40 names are easy to memorize.`, zh: `繼續練下去，訓練損失一直下降（降到 ${last(FT.full.train).toFixed(2)}），但保留集損失在第 ${BEST.full.step} 步後掉頭，一路爬到 ${last(FT.full.val).toFixed(2)}。這就是<b>過度擬合</b>：40 個名字太容易背了。` },
        { en: `You can see it in the names: ${copied.length} of its 8 names at step ${STEPS} (${copied.join(', ')}) are copied straight from the 40. So we stop early, at the lowest held-out loss.`, zh: `從名字也看得出來：第 ${STEPS} 步時它寫的 8 個名字裡，有 ${copied.length} 個（${copied.join('、')}）是直接從那 40 個名字抄來的。所以我們要及早停止，停在保留集損失最低的地方。` },
      ],
      callout: { kind: 'note', text: { en: 'The from-scratch model overfits too: its held-out loss is lowest at step 24 and then rises.', zh: '從頭訓練的模型也會過度擬合：它的保留集損失在第 24 步最低，之後就上升。' } },
    },
    {
      short: { en: 'Samples', zh: '名字範例' }, chapter: 'technique',
      title: { en: 'Read what each model writes', zh: '讀讀看每個模型寫了什麼' },
      body: [
        { en: 'Same random seed, three models. The pretrained model writes all kinds of endings. The fine-tuned one writes names ending in “a” that still look like names.', zh: '同一個亂數種子，三個模型。預訓練模型寫出各式各樣的結尾；微調後的模型寫出以「a」結尾、而且仍然像名字的名字。' },
        { en: 'The from-scratch model also ends in “a”, but produces odd ones like <code>seictra</code> and <code>narja</code>: it never learned the rest of what makes a name.', zh: '從頭訓練的模型也以「a」結尾，卻寫出 <code>seictra</code>、<code>narja</code> 這種怪名字：它從沒學過名字其他方面的規律。' },
      ],
      callout: { kind: 'note', text: { en: `These samples were written after all ${STEPS} steps, past the best point, so some are memorized.`, zh: `這些範例是跑完全部 ${STEPS} 步之後寫的，已經超過最佳點，所以有些是背下來的。` } },
    },
    {
      short: { en: 'Chat assistants', zh: '聊天助理' }, chapter: 'technique',
      title: { en: 'The same recipe makes chat assistants', zh: '聊天助理也是同一套做法' },
      body: [
        { en: 'A chat assistant starts as a model <b>pretrained</b> on a huge amount of text to predict the next token. That model knows a lot of language, but it just continues text.', zh: '聊天助理一開始，是一個用海量文字<b>預訓練</b>、學會預測下一個詞元的模型。這個模型懂很多語言知識，但它只會把文字接下去。' },
        { en: 'It is then <b>fine-tuned</b> on a much smaller set of example conversations written by people, and often further trained with human feedback. InstructGPT (2022) did this to GPT-3.', zh: '接著，再用一份小得多、由人撰寫的對話範例來<b>微調</b>它，通常還會再用人類的回饋進一步訓練。InstructGPT（2022）就是這樣改造 GPT-3 的。' },
      ],
      callout: { kind: 'note', text: { en: 'In the InstructGPT paper, people preferred the answers of a 1.3-billion-weight fine-tuned model over those of the 175-billion-weight GPT-3.', zh: '在 InstructGPT 的論文中，人們比較喜歡一個 13 億權重、經過微調的模型的回答，勝過 1,750 億權重的 GPT-3。' } },
    },
    {
      short: { en: 'Recap', zh: '回顧' }, chapter: 'recap',
      title: { en: 'Learn broadly, then specialize', zh: '先廣泛學習，再專精' },
      body: [
        { en: `<b>Pretraining</b>: lots of general data, learns how names (or language) work.<br><b>Fine-tuning</b>: continue on a small, specific set: “a” endings ${pct(FT.baseEndsA)} → ${pct(FT.full.endsA)}.<br><b>Better than scratch</b>: held-out loss ${BEST.full.loss.toFixed(2)} vs ${BEST.scratch.loss.toFixed(2)}.<br><b>Stop early</b>: small sets are memorized fast.`, zh: `<b>預訓練</b>：大量通用資料，學會名字（或語言）的規律。<br><b>微調</b>：在一小份特定資料上繼續訓練：以「a」結尾 ${pct(FT.baseEndsA)} → ${pct(FT.full.endsA)}。<br><b>勝過從頭訓練</b>：保留集損失 ${BEST.full.loss.toFixed(2)} 對 ${BEST.scratch.loss.toFixed(2)}。<br><b>及早停止</b>：小資料很快就會被背下來。` },
      ],
      callout: { kind: 'next', text: { en: 'Next: LoRA. Fine-tuning changed all 3,115 weights. Could a small add-on do the same job?', zh: '下一課：LoRA。這次微調改動了全部 3,115 個權重。能不能只靠一個小小的外掛做到同樣的事？' } },
    },
  ],

  scene(step) {
    const items: Item[] = [];
    const endsA = (n: string) => n.endsWith('a');

    const sampleRow = (k: string, names: string[], y: number, label?: string, tone: Tone = 'a') => {
      if (label) items.push({ k: k + 'l', t: 'text', x: 40, y: y - 10, text: label, size: 14, tone: 'muted' });
      names.forEach((n, i) => items.push({ k: k + i, t: 'box', x: 40 + i * 104, y, w: 96, h: 40, label: n, tone: endsA(n) ? tone : 'ink' }));
    };
    const endsBars = (shown: ('base' | 'full' | 'scratch')[], x: number, y: number, w: number, h: number) => {
      const lab = { base: t({ en: 'pretrained', zh: '預訓練' }), full: t({ en: 'fine-tuned', zh: '微調後' }), scratch: t({ en: 'scratch', zh: '從頭訓練' }) };
      const v = { base: FT.baseEndsA, full: FT.full.endsA, scratch: FT.scratch.endsA };
      items.push({ k: 'ebar', t: 'bars', x, y, w, h, items: shown.map((s) => ({ label: lab[s], v: v[s], hi: s === 'full' })), title: t({ en: 'names ending in “a”', zh: '以「a」結尾的名字' }) });
    };
    const chart = (series: { pts: number[]; tone: Tone }[], ymin = 0.5, ymax = 3.5) => {
      items.push({
        k: 'chart', t: 'chart', x: 80, y: 50, w: 480, h: 250, ymin, ymax, xmax: STEPS, every: FT.full.every, series,
        xlabel: t({ en: 'fine-tuning steps', zh: '微調步數' }), ylabel: t({ en: 'loss on held-out names ending in “a”', zh: '以「a」結尾的保留名字上的損失' }),
        guide: { y: FT.baseValLoss, label: t({ en: `pretrained · ${FT.baseValLoss.toFixed(2)}`, zh: `預訓練 · ${FT.baseValLoss.toFixed(2)}` }) },
      });
    };
    const bestMark = (k: string, b: { loss: number; step: number }, tone: Tone, ymin = 0.5, ymax = 3.5) => {
      const x = 80 + (b.step / STEPS) * 480, y = 50 + 250 - ((b.loss - ymin) / (ymax - ymin)) * 250;
      items.push({ k, t: 'arrow', x1: x, y1: y + 40, x2: x, y2: y + 6, tone });
      items.push({ k: k + 't', t: 'text', x, y: y + 56, text: t({ en: `best ${b.loss.toFixed(2)} @ ${b.step}`, zh: `最佳 ${b.loss.toFixed(2)} @ ${b.step}` }), size: 13, anchor: 'middle', tone, bold: true });
    };

    let ends = pct(FT.baseEndsA), loss = FT.baseValLoss.toFixed(2), data = '—', steps = '0';

    if (step === 0) {
      sampleRow('s', SAMPLES.base, 90, t({ en: 'the pretrained model writes (names ending in “a” in teal):', zh: '預訓練模型寫出（以「a」結尾的標為青綠色）：' }));
      endsBars(['base'], 330, 220, 240, 230);
    }

    if (step === 1) {
      items.push({ k: 'pd', t: 'box', x: 60, y: 80, w: 250, h: 70, label: t({ en: `${fmt(TRAIN_NAMES.length)} training names`, zh: `${fmt(TRAIN_NAMES.length)} 個訓練名字` }), sub: t({ en: 'all kinds of names', zh: '各式各樣的名字' }) });
      items.push({ k: 'pa', t: 'arrow', x1: 316, y1: 115, x2: 384, y2: 115 });
      items.push({ k: 'pt', t: 'box', x: 390, y: 80, w: 180, h: 70, label: t({ en: 'pretraining', zh: '預訓練' }), sub: t({ en: '6,000 steps', zh: '6,000 步' }) });
      items.push({ k: 'pa2', t: 'arrow', x1: 576, y1: 115, x2: 644, y2: 115 });
      items.push({ k: 'pm', t: 'box', x: 650, y: 80, w: 200, h: 70, label: t({ en: 'pretrained model', zh: '預訓練模型' }), sub: t({ en: '3,115 weights', zh: '3,115 個權重' }), tone: 'a' });
      const knows = [
        t({ en: '· which letters follow which', zh: '· 哪些字母會接在哪些後面' }),
        t({ en: '· how long a name usually is', zh: '· 名字通常多長' }),
        t({ en: '· how names usually end', zh: '· 名字通常怎麼結尾' }),
      ];
      items.push({ k: 'pk', t: 'text', x: 650, y: 190, text: t({ en: 'it already knows:', zh: '它已經知道：' }), size: 14, tone: 'muted' });
      knows.forEach((s, i) => items.push({ k: 'pk' + i, t: 'text', x: 650, y: 218 + i * 26, text: s, size: 14 }));
      sampleRow('s', SAMPLES.base, 360, t({ en: 'it writes:', zh: '它寫出：' }));
      data = fmt(TRAIN_NAMES.length);
    }

    if (step === 2) {
      items.push({ k: 'pm', t: 'box', x: 40, y: 30, w: 200, h: 56, label: t({ en: 'pretrained model', zh: '預訓練模型' }), sub: t({ en: 'starting point', zh: '起點' }), tone: 'a' });
      items.push({ k: 'pa2', t: 'arrow', x1: 246, y1: 58, x2: 314, y2: 58, tone: 'a' });
      items.push({ k: 'pt', t: 'box', x: 320, y: 30, w: 260, h: 56, label: t({ en: 'fine-tuning', zh: '微調' }), sub: t({ en: 'learning rate 0.003', zh: '學習率 0.003' }), tone: 'b' });
      items.push({ k: 'fl', t: 'text', x: 50, y: 130, text: t({ en: `the ${FT.trainNames} fine-tuning names (all from the training set)`, zh: `${FT.trainNames} 個微調用的名字（都來自訓練集）` }), size: 14, tone: 'muted' });
      FT_NAMES.forEach((n, i) => items.push({ k: 'fn' + i, t: 'box', x: 50 + (i % 8) * 102, y: 145 + Math.floor(i / 8) * 46, w: 94, h: 38, label: n, tone: 'b' }));
      items.push({ k: 'fs', t: 'text', x: 450, y: 400, text: t({ en: `${FT.trainNames} names = ${(100 * FT.trainNames / TRAIN_NAMES.length).toFixed(1)}% of the pretraining data`, zh: `${FT.trainNames} 個名字 = 預訓練資料的 ${(100 * FT.trainNames / TRAIN_NAMES.length).toFixed(1)}%` }), size: 16, bold: true, anchor: 'middle' });
      data = String(FT.trainNames);
    }

    if (step === 3 || step === 4) {
      const series: { pts: number[]; tone: Tone }[] = [{ pts: FT.full.val, tone: 'a' }];
      if (step === 4) series.unshift({ pts: FT.scratch.val, tone: 'b' });
      chart(series, 1.5, 3.5);
      bestMark('bm', BEST.full, 'a', 1.5, 3.5);
      if (step === 4) bestMark('bs', BEST.scratch, 'b', 1.5, 3.5);
      endsBars(step === 3 ? ['base', 'full'] : ['base', 'full', 'scratch'], 620, 60, 250, 250);
      sampleRow('s', step === 3 ? SAMPLES.full : SAMPLES.scratch, 410, step === 3
        ? t({ en: `fine-tuned model, after ${STEPS} steps:`, zh: `微調後的模型，跑完 ${STEPS} 步：` })
        : t({ en: `from-scratch model, after ${STEPS} steps:`, zh: `從頭訓練的模型，跑完 ${STEPS} 步：` }), step === 3 ? 'a' : 'b');
      ends = pct(step === 3 ? FT.full.endsA : FT.scratch.endsA);
      loss = (step === 3 ? BEST.full.loss : BEST.scratch.loss).toFixed(2);
      data = String(FT.trainNames);
      steps = String(step === 3 ? BEST.full.step : BEST.scratch.step);
    }

    if (step === 5) {
      chart([{ pts: FT.full.train, tone: 'muted' }, { pts: FT.full.val, tone: 'a' }], 0.5, 2.5);
      bestMark('bm', BEST.full, 'a', 0.5, 2.5);
      items.push({ k: 'stop', t: 'box', x: 620, y: 70, w: 250, h: 70, label: t({ en: `stop at step ${BEST.full.step}`, zh: `在第 ${BEST.full.step} 步停下` }), sub: t({ en: `held-out ${BEST.full.loss.toFixed(2)}`, zh: `保留集 ${BEST.full.loss.toFixed(2)}` }), tone: 'a' });
      items.push({ k: 'over', t: 'box', x: 620, y: 170, w: 250, h: 70, label: t({ en: `step ${STEPS}: held-out ${last(FT.full.val).toFixed(2)}`, zh: `第 ${STEPS} 步：保留集 ${last(FT.full.val).toFixed(2)}` }), sub: t({ en: `training loss ${last(FT.full.train).toFixed(2)}`, zh: `訓練損失 ${last(FT.full.train).toFixed(2)}` }), tone: 'bad' });
      items.push({ k: 'sl', t: 'text', x: 40, y: 400, text: t({ en: `fine-tuned at step ${STEPS} · red = copied from the 40 training names`, zh: `第 ${STEPS} 步的微調模型 · 紅色 = 直接抄自 40 個訓練名字` }), size: 14, tone: 'muted' });
      SAMPLES.full.forEach((n, i) => items.push({ k: 's' + i, t: 'box', x: 40 + i * 104, y: 410, w: 96, h: 40, label: n, tone: FT_NAMES.includes(n) ? 'bad' : 'ink' }));
      ends = pct(FT.full.endsA); loss = last(FT.full.val).toFixed(2); data = String(FT.trainNames); steps = String(STEPS);
    }

    if (step === 6) {
      sampleRow('b', SAMPLES.base, 70, t({ en: `pretrained · ${pct(FT.baseEndsA)} end in “a”`, zh: `預訓練 · ${pct(FT.baseEndsA)} 以「a」結尾` }));
      sampleRow('s', SAMPLES.full, 200, t({ en: `fine-tuned · ${pct(FT.full.endsA)} end in “a”`, zh: `微調後 · ${pct(FT.full.endsA)} 以「a」結尾` }));
      sampleRow('c', SAMPLES.scratch, 330, t({ en: `from scratch · ${pct(FT.scratch.endsA)} end in “a”`, zh: `從頭訓練 · ${pct(FT.scratch.endsA)} 以「a」結尾` }), 'b');
      items.push({ k: 'sn', t: 'text', x: 450, y: 450, text: t({ en: 'same seed for all three · 400 samples each for the percentages', zh: '三者使用同一個亂數種子 · 百分比各用 400 個名字計算' }), size: 14, tone: 'muted', anchor: 'middle' });
      ends = pct(FT.full.endsA); data = String(FT.trainNames); steps = String(STEPS); loss = last(FT.full.val).toFixed(2);
    }

    if (step === 7) {
      const row = (k: string, y: number, a: string, as: string, b: string, bs: string, c: string, tone: Tone) => {
        items.push({ k: k + '0', t: 'box', x: 40, y, w: 240, h: 70, label: a, sub: as });
        items.push({ k: k + 'a', t: 'arrow', x1: 286, y1: y + 35, x2: 334, y2: y + 35 });
        items.push({ k: k + '1', t: 'box', x: 340, y, w: 240, h: 70, label: b, sub: bs, tone: 'b' });
        items.push({ k: k + 'b', t: 'arrow', x1: 586, y1: y + 35, x2: 634, y2: y + 35 });
        items.push({ k: k + '2', t: 'box', x: 640, y, w: 220, h: 70, label: c, tone });
      };
      items.push({ k: 'h1', t: 'text', x: 40, y: 70, text: t({ en: 'our names (real runs)', zh: '我們的名字（真實訓練）' }), size: 14, tone: 'muted' });
      row('o', 84, t({ en: 'pretrain', zh: '預訓練' }), t({ en: `${fmt(TRAIN_NAMES.length)} names`, zh: `${fmt(TRAIN_NAMES.length)} 個名字` }), t({ en: 'fine-tune', zh: '微調' }), t({ en: `${FT.trainNames} names ending in “a”`, zh: `${FT.trainNames} 個以「a」結尾的名字` }), t({ en: '“a” names', zh: '「a」結尾的名字' }), 'a');
      items.push({ k: 'h2', t: 'text', x: 40, y: 250, text: t({ en: 'a chat assistant (general outline)', zh: '聊天助理（大致流程）' }), size: 14, tone: 'muted' });
      row('c', 264, t({ en: 'pretrain', zh: '預訓練' }), t({ en: 'a huge amount of text', zh: '海量文字' }), t({ en: 'fine-tune', zh: '微調' }), t({ en: 'example conversations', zh: '對話範例' }), t({ en: 'assistant', zh: '助理' }), 'a');
      items.push({ k: 'cn', t: 'text', x: 450, y: 420, text: t({ en: 'expensive and general → cheap and specific', zh: '昂貴且通用 → 便宜且專門' }), size: 16, bold: true, anchor: 'middle' });
      ends = pct(FT.full.endsA); loss = BEST.full.loss.toFixed(2); data = String(FT.trainNames); steps = String(BEST.full.step);
    }

    if (step === 8) {
      chart([{ pts: FT.scratch.val, tone: 'b' }, { pts: FT.full.val, tone: 'a' }], 1.5, 3.5);
      bestMark('bm', BEST.full, 'a', 1.5, 3.5);
      bestMark('bs', BEST.scratch, 'b', 1.5, 3.5);
      endsBars(['base', 'full', 'scratch'], 620, 60, 250, 250);
      sampleRow('s', SAMPLES.full, 410, t({ en: 'fine-tuned model writes:', zh: '微調後的模型寫出：' }));
      ends = pct(FT.full.endsA); loss = BEST.full.loss.toFixed(2); data = String(FT.trainNames); steps = String(BEST.full.step);
    }

    return {
      metrics: [
        { label: { en: 'Ends in “a”', zh: '以「a」結尾' }, value: ends },
        { label: { en: 'Held-out loss', zh: '保留集損失' }, value: loss },
        { label: { en: 'Training names', zh: '訓練名字數' }, value: data },
        { label: { en: 'Fine-tuning steps', zh: '微調步數' }, value: steps },
      ],
      items,
      legend: step === 3 || step === 4 || step === 8
        ? [{ tone: 'a', text: { en: 'pretrained + fine-tuned', zh: '預訓練 + 微調' } }, ...(step !== 3 ? [{ tone: 'b' as const, text: { en: 'from scratch on the 40 names', zh: '只用 40 個名字從頭訓練' } }] : [])]
        : step === 5 ? [{ tone: 'a', text: { en: 'held-out loss (the gray line is the training loss)', zh: '保留集損失（灰線是訓練損失）' } }] : undefined,
    };
  },
};
