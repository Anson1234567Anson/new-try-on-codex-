import './style.css';
import { UI, fill, getLang, setLang, t, type Lang } from './i18n';
import { BUILT_ORDER, LESSONS, MODULES, TOTAL } from './course';
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

function renderScene(): void {
  const sc = lesson().scene(step);
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
  const more = lesson().tick?.(step) ?? false;
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

document.addEventListener('keydown', (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'ArrowRight') go(1);
  else if (e.key === 'ArrowLeft') go(-1);
  else if (e.key === 'r' || e.key === 'R') replay();
  else if (e.key === ' ') { e.preventDefault(); togglePause(); }
});

window.addEventListener('hashchange', route);
route();
