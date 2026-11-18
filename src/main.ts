import './style.css';
import { UI, fill, getLang, setLang, t, type Lang } from './i18n';
import { BUILT_ORDER, LESSONS, MODULES, TOTAL } from './course';
import type { Control, Input } from './lessons/types';
import { Stage } from './engine/scene';

// ---------- saved state ----------
const store = {
  get<T>(key: string, fallback: T): T {
    try { const v = localStorage.getItem('tfl:' + key); return v === null ? fallback : (JSON.parse(v) as T); } catch { return fallback; }
  },
  set(key: string, value: unknown): void {
    try { localStorage.setItem('tfl:' + key, JSON.stringify(value)); } catch { /* private mode: progress just is not kept */ }
  },
};

let done: string[] = store.get('done', []);
let theme: string = (() => { try { return localStorage.getItem('tfl:theme') ?? 'light'; } catch { return 'light'; } })();
setLang(store.get<Lang>('lang', navigator.language.toLowerCase().startsWith('zh') ? 'zh' : 'en'));

let lessonId = BUILT_ORDER[0];
let step = 0;
let paused = false;
let raf = 0;


// ---------- page skeleton ----------
const app = document.getElementById('app')!;
app.innerHTML = `
  <header class="top">
    <a class="brand" href="#/"><span class="brand__mark">T</span><span id="brand"></span></a>
    <div class="top__right">
      <div class="seg" role="group" aria-label="Language">
        <button data-lang="en">EN</button><button data-lang="zh">繁中</button>
      </div>
      <button class="icon-btn" data-act="theme" id="theme" aria-label="theme">◐</button>
    </div>
  </header>
  <div class="layout">
    <aside class="side" id="side"></aside>
    <main class="center">
      <section class="card head" id="head"></section>
      <section class="card stage">
        <div class="stage__bar">
          <div class="metrics" id="metrics"></div>
          <div class="stage__btns" id="btns"></div>
        </div>
        <div class="controls" id="controls" hidden></div>
        <div class="stage__scroll"><svg id="svg" class="stage__svg" role="img"></svg></div>
        <div class="stage__foot"><div class="legend" id="legend"></div><div class="real" id="real"></div></div>
      </section>
      <section class="card timeline" id="timeline"></section>
    </main>
    <aside class="panel">
      <section class="card step" id="step"></section>
      <section class="card learn" id="learn"></section>
    </aside>
  </div>`;

const $ = (id: string) => document.getElementById(id)!;
const stage = new Stage($('svg') as unknown as SVGSVGElement);
const lesson = () => LESSONS[lessonId];

// ---------- rendering ----------
function renderSide(): void {
  const doneCount = done.filter((id) => id in LESSONS).length;
  $('side').innerHTML = `
    <div class="side__title"><b>${t(UI.course)}</b><span>${TOTAL} ${t(UI.lessons)}</span></div>
    ${MODULES.map((m, mi) => `
      <div class="mod">
        <div class="mod__title"><span class="mod__n">${String(mi + 1).padStart(2, '0')}</span>${t(m.title)}</div>
        ${m.lessons.map((l) => l.id in LESSONS
          ? `<a class="lesson ${l.id === lessonId ? 'is-current' : ''}" href="#/${l.id}/1"><span class="lesson__dot ${done.includes(l.id) ? 'is-done' : ''}">${done.includes(l.id) ? '✓' : ''}</span>${t(l.title)}</a>`
          : `<div class="lesson is-soon"><span class="lesson__dot"></span>${t(l.title)}<em>${t(UI.soon)}</em></div>`).join('')}
      </div>`).join('')}
    <div class="side__progress">
      <div><b>${t(UI.progress)}</b><span>${doneCount} / ${TOTAL}</span></div>
      <div class="bar"><i style="width:${(doneCount / TOTAL) * 100}%"></i></div>
    </div>`;
}

function renderHead(): void {
  const l = lesson();
  const mi = MODULES.findIndex((m) => m.lessons.some((x) => x.id === l.id));
  const li = MODULES[mi].lessons.findIndex((x) => x.id === l.id);
  $('head').innerHTML = `
    <div class="crumb">${String(mi + 1).padStart(2, '0')}. ${t(MODULES[mi].title)} <span>/</span> ${fill(UI.lessonOf, { a: li + 1, b: MODULES[mi].lessons.length })}</div>
    <div class="head__row">
      <h1>${t(l.title)}</h1>
      <div class="chips">
        <span class="chip">${l.minutes} ${t(UI.min)}</span>
        <span class="chip chip--b">${t(UI.level[l.level])}</span>
        <span class="chip chip--a">${l.steps.length} ${t(UI.steps)}</span>
      </div>
    </div>
    <p class="blurb">${t(l.blurb)}</p>`;
}

// control values the learner has set, per lesson step ("lesson/step/key")
const inputs = new Map<string, number | string | boolean>();
const inputFor = (controls: Control[] | undefined): Input => {
  const out: Input = {};
  for (const c of controls ?? []) out[c.key] = inputs.get(`${lessonId}/${step}/${c.key}`) ?? c.value;
  return out;
};
let controlsSig = '';

function renderControls(controls: Control[] | undefined, input: Input): void {
  const box = $('controls');
  const sig = `${lessonId}/${step}/${getLang()}/` + (controls ?? []).map((c) => c.key + c.kind).join(',');
  if (!controls?.length) { box.hidden = true; box.innerHTML = ''; controlsSig = sig; return; }
  box.hidden = false;
  if (sig !== controlsSig) {
    // build the controls once per step, so typing and dragging are not interrupted
    controlsSig = sig;
    box.innerHTML = controls.map((c) => {
      const id = `ctl-${c.key}`;
      const label = `<label for="${id}">${t(c.label)}</label>`;
      if (c.kind === 'slider') return `<div class="ctl ctl--slider">${label}<input id="${id}" data-key="${c.key}" type="range" min="${c.min}" max="${c.max}" step="${c.step}" value="${input[c.key]}"><output id="${id}-out"></output></div>`;
      if (c.kind === 'text') return `<div class="ctl ctl--text">${label}<input id="${id}" data-key="${c.key}" type="text" autocomplete="off" spellcheck="false" ${c.maxLength ? `maxlength="${c.maxLength}"` : ''} placeholder="${c.placeholder ? t(c.placeholder) : ''}" value="${String(input[c.key])}"></div>`;
      if (c.kind === 'toggle') return `<div class="ctl ctl--toggle"><input id="${id}" data-key="${c.key}" type="checkbox" ${input[c.key] ? 'checked' : ''}>${label}</div>`;
      return `<div class="ctl ctl--choice">${label}<div class="seg">${c.options.map((o) => `<button type="button" data-key="${c.key}" data-value="${o.value}">${t(o.label)}</button>`).join('')}</div></div>`;
    }).join('');
  }
  for (const c of controls) {
    if (c.kind === 'slider') {
      const out = document.getElementById(`ctl-${c.key}-out`);
      if (out) out.textContent = c.format ? c.format(input[c.key] as number) : String(input[c.key]);
    }
    if (c.kind === 'choice') box.querySelectorAll<HTMLButtonElement>(`[data-key="${c.key}"]`).forEach((b) => b.classList.toggle('is-on', b.dataset.value === input[c.key]));
  }
}

function setInput(key: string, value: number | string | boolean): void {
  inputs.set(`${lessonId}/${step}/${key}`, value);
  if (lesson().tick) { paused = false; kick(); } else renderScene();
}

function renderScene(): void {
  const l = lesson();
  // first pass learns which controls this step has, second pass draws with their values
  const probe = l.scene(step, {});
  const input = inputFor(probe.controls);
  const sc = probe.controls?.length ? l.scene(step, input) : probe;
  renderControls(sc.controls, input);
  $('metrics').innerHTML = sc.metrics.map((m) => `<div class="metric"><span>${t(m.label)}</span><b>${m.value}</b></div>`).join('');
  $('legend').innerHTML = (sc.legend ?? []).map((g) => `<span><i class="lg lg--${g.tone}"></i>${t(g.text)}</span>`).join('');
  stage.render(sc.items);
}

function renderChrome(): void {
  const l = lesson();
  const s = l.steps[step];
  const last = step === l.steps.length - 1;
  const nextLesson = BUILT_ORDER[BUILT_ORDER.indexOf(lessonId) + 1];

  $('btns').innerHTML = `
    <button class="btn" data-act="replay">↺ ${t(UI.replay)}</button>
    ${l.tick ? `<button class="btn" data-act="pause">${paused ? '▶ ' + t(UI.resume) : '❚❚ ' + t(UI.pause)}</button>` : ''}`;
  $('real').textContent = t(UI.realNumbers);

  $('timeline').innerHTML = `
    <div class="timeline__head"><b>${t(UI.inLesson)}</b><span>${fill(UI.stepOf, { a: step + 1, b: l.steps.length })}</span></div>
    <ol class="dots">${l.steps.map((x, i) => `
      <li class="${i < step ? 'is-past' : i === step ? 'is-now' : ''}">
        <a href="#/${l.id}/${i + 1}"><span>${i + 1}</span><em>${t(x.short)}</em></a>
      </li>`).join('')}</ol>`;

  $('step').innerHTML = `
    <div class="step__top"><span class="step__n">${fill(UI.stepOf, { a: step + 1, b: l.steps.length })}</span><span class="tag tag--${s.chapter}">${t(UI.chapter[s.chapter])}</span></div>
    <h2>${t(s.title)}</h2>
    ${s.body.map((p) => `<p>${t(p)}</p>`).join('')}
    ${s.callout ? `<div class="callout callout--${s.callout.kind}"><b>${t(UI.callout[s.callout.kind])}</b><p>${t(s.callout.text)}</p></div>` : ''}
    ${s.changed ? `<div class="changed"><b>${t(UI.changed)}</b>${s.changed.map((c) => `<div><i class="sign sign--${c.sign === '+' ? 'add' : 'mod'}">${c.sign}</i>${t(c.text)}</div>`).join('')}</div>` : ''}
    <div class="nav">
      <a class="btn ${step === 0 ? 'is-off' : ''}" href="#/${l.id}/${Math.max(1, step)}">← ${t(UI.back)}</a>
      ${last
        ? nextLesson
          ? `<a class="btn btn--go" href="#/${nextLesson}/1">${t(LESSONS[nextLesson].title)} →</a>`
          : `<span class="btn btn--go is-off">✓ ${t(UI.finish)}</span>`
        : `<a class="btn btn--go" href="#/${l.id}/${step + 2}">${t(UI.next)}: ${t(l.steps[step + 1].short)} →</a>`}
    </div>
    <div class="keys">${t(UI.keys)}</div>`;

  $('learn').innerHTML = `<b>${t(UI.learn)}</b>${l.learn.map((x) => `<div class="tick ${step >= x.step ? 'is-on' : ''}"><i>${step >= x.step ? '✓' : ''}</i>${t(x.text)}</div>`).join('')}`;
}

function renderAll(): void {
  document.documentElement.lang = getLang() === 'zh' ? 'zh-Hant-TW' : 'en';
  document.title = `${t(lesson().title)} · ${t(UI.brand)}`;
  $('brand').textContent = t(UI.brand);
  $('theme').title = t(UI.theme);
  document.querySelectorAll<HTMLButtonElement>('[data-lang]').forEach((b) => b.classList.toggle('is-on', b.dataset.lang === getLang()));
  renderSide();
  renderHead();
  renderChrome();
  renderScene();
}

// ---------- live lessons ----------
function loop(): void {
  raf = 0;
  if (paused) return;
  const l = lesson();
  const more = l.tick?.(step, inputFor(l.scene(step, {}).controls)) ?? false;
  renderScene();
  if (more) raf = requestAnimationFrame(loop);
}
function kick(): void {
  if (raf) cancelAnimationFrame(raf);
  raf = 0;
  if (lesson().tick) loop(); else renderScene();
}

// ---------- navigation ----------
function route(): void {
  const [, id, n] = location.hash.split('/');
  const wanted = id && id in LESSONS ? id : store.get<{ id: string; step: number } | null>('last', null)?.id ?? BUILT_ORDER[0];
  const nextId = wanted in LESSONS ? wanted : BUILT_ORDER[0];
  const max = LESSONS[nextId].steps.length;
  const nextStep = id === nextId ? Math.min(max, Math.max(1, parseInt(n) || 1)) - 1 : id ? 0 : store.get<{ id: string; step: number } | null>('last', null)?.step ?? 0;
  if (nextId !== lessonId) stage.clear();
  lessonId = nextId;
  step = Math.min(max - 1, nextStep);
  paused = false;
  store.set('last', { id: lessonId, step });
  if (step === max - 1 && !done.includes(lessonId)) { done = [...done, lessonId]; store.set('done', done); }
  renderAll();
  kick();
}

function go(delta: number): void {
  const n = Math.min(lesson().steps.length - 1, Math.max(0, step + delta));
  if (n !== step) location.hash = `#/${lessonId}/${n + 1}`;
}

function replay(): void {
  lesson().reset?.();
  stage.clear();
  paused = false;
  renderChrome();
  kick();
}

function togglePause(): void {
  if (!lesson().tick) return;
  paused = !paused;
  renderChrome();
  if (!paused) kick();
}

document.addEventListener('click', (e) => {
  const el = (e.target as HTMLElement).closest<HTMLElement>('[data-act],[data-lang]');
  if (!el) return;
  if (el.dataset.lang) { setLang(el.dataset.lang as Lang); store.set('lang', getLang()); renderAll(); return; }
  if (el.dataset.act === 'replay') replay();
  if (el.dataset.act === 'pause') togglePause();
  if (el.dataset.act === 'theme') {
    theme = theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = theme;
    try { localStorage.setItem('tfl:theme', theme); } catch { /* not kept */ }
  }
});

$('controls').addEventListener('input', (e) => {
  const el = e.target as HTMLInputElement;
  const key = el.dataset.key;
  if (!key) return;
  if (el.type === 'range') setInput(key, parseFloat(el.value));
  else if (el.type === 'checkbox') setInput(key, el.checked);
  else {
    const c = lesson().scene(step, {}).controls?.find((x) => x.key === key);
    let v = el.value;
    if (c && c.kind === 'text' && c.pattern) v = v.replace(new RegExp(`[^${c.pattern}]`, 'g'), '');
    if (v !== el.value) el.value = v;
    setInput(key, v);
  }
});
$('controls').addEventListener('click', (e) => {
  const b = (e.target as HTMLElement).closest<HTMLButtonElement>('button[data-key]');
  if (b?.dataset.key && b.dataset.value !== undefined) setInput(b.dataset.key, b.dataset.value);
});

document.addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  // typing in a control must not step the lesson
  const tag = (e.target as HTMLElement).tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  if (e.key === 'ArrowRight') go(1);
  else if (e.key === 'ArrowLeft') go(-1);
  else if (e.key === 'r' || e.key === 'R') replay();
  else if (e.key === ' ') { e.preventDefault(); togglePause(); }
});

window.addEventListener('hashchange', route);
route();
