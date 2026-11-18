// The diagram engine. A lesson describes each step as a list of keyed items;
// the stage keeps items that survive from one step to the next and animates
// them to their new place, so the picture evolves instead of being replaced.

export type Tone = 'ink' | 'muted' | 'a' | 'b' | 'good' | 'bad';

export type Item =
  | { k: string; t: 'token'; x: number; y: number; text: string; sub?: string; tone?: Tone; dim?: boolean }
  | { k: string; t: 'vec'; x: number; y: number; v: number[]; cell?: number; max?: number; dim?: boolean; label?: string }
  | { k: string; t: 'heat'; x: number; y: number; m: (number | null)[][]; rows: string[]; cols: string[]; cell?: number; hi?: number; fmt?: 'pct' | 'num'; max?: number }
  | { k: string; t: 'text'; x: number; y: number; text: string; size?: number; tone?: Tone; anchor?: 'start' | 'middle' | 'end'; bold?: boolean; mono?: boolean }
  | { k: string; t: 'arrow'; x1: number; y1: number; x2: number; y2: number; tone?: Tone; dashed?: boolean }
  | { k: string; t: 'box'; x: number; y: number; w: number; h: number; label?: string; sub?: string; tone?: Tone; active?: boolean }
  | { k: string; t: 'chart'; x: number; y: number; w: number; h: number; series: { pts: number[]; tone: Tone }[]; ymin: number; ymax: number; xmax: number; every: number; xlabel: string; ylabel: string; guide?: { y: number; label: string } }
  | { k: string; t: 'bars'; x: number; y: number; w: number; h: number; items: { label: string; v: number; hi?: boolean }[]; title?: string };

export const W = 900;
export const H = 520;
export const TOKEN = 44; // token chip size

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const n1 = (v: number) => +v.toFixed(1);

function inner(it: Item): string {
  switch (it.t) {
    case 'token': {
      const sub = it.sub ? `<text class="s-sub" x="${TOKEN / 2}" y="${TOKEN + 15}" text-anchor="middle">${esc(it.sub)}</text>` : '';
      return `<rect class="s-token tone-${it.tone ?? 'ink'}" width="${TOKEN}" height="${TOKEN}" rx="9"/>
        <text class="s-token-text" x="${TOKEN / 2}" y="${TOKEN / 2 + 7}" text-anchor="middle">${esc(it.text)}</text>${sub}`;
    }
    case 'vec': {
      const cell = it.cell ?? 11;
      const max = it.max ?? Math.max(0.5, ...it.v.map(Math.abs));
      const cells = it.v.map((v, i) =>
        `<rect x="${i * cell}" width="${cell - 1.5}" height="22" rx="2" class="${v >= 0 ? 's-pos' : 's-neg'}" fill-opacity="${n1(Math.min(1, Math.abs(v) / max) * 0.92 + 0.08)}"><title>${v.toFixed(3)}</title></rect>`).join('');
      const label = it.label ? `<text class="s-sub" x="0" y="-6">${esc(it.label)}</text>` : '';
      return `<rect class="s-vec-bg" x="-3" y="-3" width="${it.v.length * cell + 4.5}" height="28" rx="5"/>${cells}${label}`;
    }
    case 'heat': {
      const cell = it.cell ?? 52;
      const max = it.max ?? 1;
      let s = '';
      it.cols.forEach((c, j) => { s += `<text class="s-axis" x="${j * cell + cell / 2}" y="-10" text-anchor="middle">${esc(c)}</text>`; });
      it.rows.forEach((r, i) => {
        s += `<text class="s-axis ${it.hi === i ? 's-axis-hi' : ''}" x="-12" y="${i * cell + cell / 2 + 5}" text-anchor="end">${esc(r)}</text>`;
        it.m[i].forEach((v, j) => {
          const x = j * cell, y = i * cell;
          if (v === null) {
            s += `<rect class="s-cell-masked" x="${x}" y="${y}" width="${cell - 3}" height="${cell - 3}" rx="5"/>
              <text class="s-cell-x" x="${x + cell / 2 - 1.5}" y="${y + cell / 2 + 4}" text-anchor="middle">×</text>`;
          } else {
            const shade = it.fmt === 'pct' ? v / max : Math.abs(v) / max;
            const cls = it.fmt === 'pct' ? 's-heat' : v >= 0 ? 's-pos' : 's-neg';
            const label = it.fmt === 'pct' ? Math.round(v * 100) + '%' : v.toFixed(1);
            s += `<rect class="s-cell-bg" x="${x}" y="${y}" width="${cell - 3}" height="${cell - 3}" rx="5"/>
              <rect class="${cls}" x="${x}" y="${y}" width="${cell - 3}" height="${cell - 3}" rx="5" fill-opacity="${n1(Math.min(1, shade) * 0.9)}"/>
              <text class="s-cell-text" x="${x + cell / 2 - 1.5}" y="${y + cell / 2 + 4}" text-anchor="middle">${label}</text>`;
          }
        });
        if (it.hi === i) s += `<rect class="s-row-hi" x="-4" y="${i * cell - 4}" width="${it.cols.length * cell + 5}" height="${cell + 5}" rx="8"/>`;
      });
      return s;
    }
    case 'text':
      return `<text class="s-text tone-text-${it.tone ?? 'ink'} ${it.bold ? 's-bold' : ''} ${it.mono ? 's-mono' : ''}" font-size="${it.size ?? 14}" text-anchor="${it.anchor ?? 'start'}">${esc(it.text)}</text>`;
    case 'arrow': {
      const dx = it.x2 - it.x1, dy = it.y2 - it.y1, len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len, hx = it.x2 - ux * 9, hy = it.y2 - uy * 9;
      return `<line class="s-arrow tone-line-${it.tone ?? 'muted'}" x1="${it.x1}" y1="${it.y1}" x2="${n1(hx)}" y2="${n1(hy)}" ${it.dashed ? 'stroke-dasharray="5 5"' : ''}/>
        <polygon class="tone-fill-${it.tone ?? 'muted'}" points="${it.x2},${it.y2} ${n1(hx - uy * 5)},${n1(hy + ux * 5)} ${n1(hx + uy * 5)},${n1(hy - ux * 5)}"/>`;
    }
    case 'box':
      return `<rect class="s-box tone-${it.tone ?? 'ink'} ${it.active ? 's-box-active' : ''}" width="${it.w}" height="${it.h}" rx="10"/>
        ${it.label ? `<text class="s-box-label" x="${it.w / 2}" y="${it.h / 2 + (it.sub ? -2 : 5)}" text-anchor="middle">${esc(it.label)}</text>` : ''}
        ${it.sub ? `<text class="s-sub" x="${it.w / 2}" y="${it.h / 2 + 16}" text-anchor="middle">${esc(it.sub)}</text>` : ''}`;
    case 'chart': {
      const px = (i: number) => (i / Math.max(1, it.xmax)) * it.w;
      const py = (v: number) => it.h - ((Math.min(it.ymax, Math.max(it.ymin, v)) - it.ymin) / (it.ymax - it.ymin)) * it.h;
      let s = `<rect class="s-chart-bg" width="${it.w}" height="${it.h}" rx="8"/>`;
      for (let g = 0; g <= 4; g++) {
        const v = it.ymin + ((it.ymax - it.ymin) * g) / 4;
        s += `<line class="s-grid" x1="0" x2="${it.w}" y1="${n1(py(v))}" y2="${n1(py(v))}"/>
          <text class="s-axis" x="-8" y="${n1(py(v) + 4)}" text-anchor="end">${v.toFixed(1)}</text>`;
      }
      if (it.guide) s += `<line class="s-guide" x1="0" x2="${it.w}" y1="${n1(py(it.guide.y))}" y2="${n1(py(it.guide.y))}"/>
        <text class="s-sub" x="${it.w - 6}" y="${n1(py(it.guide.y) - 6)}" text-anchor="end">${esc(it.guide.label)}</text>`;
      for (const ser of it.series) {
        if (!ser.pts.length) continue;
        const pts = ser.pts.map((v, i) => `${n1(px(i * it.every))},${n1(py(Number.isFinite(v) ? v : it.ymax))}`);
        s += `<polyline class="s-series tone-line-${ser.tone}" points="${pts.join(' ')}"/>`;
        const last = pts[pts.length - 1].split(',');
        s += `<circle class="tone-fill-${ser.tone}" cx="${last[0]}" cy="${last[1]}" r="4"/>`;
      }
      s += `<text class="s-axis" x="${it.w / 2}" y="${it.h + 22}" text-anchor="middle">${esc(it.xlabel)}</text>
        <text class="s-axis" x="0" y="-10">${esc(it.ylabel)}</text>
        <text class="s-axis" x="${it.w}" y="${it.h + 22}" text-anchor="end">${it.xmax}</text>
        <text class="s-axis" x="0" y="${it.h + 22}">0</text>`;
      return s;
    }
    case 'bars': {
      const max = Math.max(0.01, ...it.items.map((b) => b.v));
      const bw = it.w / it.items.length;
      let s = it.title ? `<text class="s-sub" x="0" y="-12">${esc(it.title)}</text>` : '';
      it.items.forEach((b, i) => {
        const bh = (b.v / max) * (it.h - 34);
        s += `<rect class="s-bar ${b.hi ? 's-bar-hi' : ''}" x="${n1(i * bw + 3)}" y="${n1(it.h - 18 - bh)}" width="${n1(bw - 6)}" height="${n1(Math.max(1, bh))}" rx="3"/>
          <text class="s-axis ${b.hi ? 's-axis-hi' : ''}" x="${n1(i * bw + bw / 2)}" y="${it.h - 2}" text-anchor="middle">${esc(b.label)}</text>
          <text class="s-bar-val" x="${n1(i * bw + bw / 2)}" y="${n1(it.h - 23 - bh)}" text-anchor="middle">${Math.round(b.v * 100)}%</text>`;
      });
      return s;
    }
  }
}

function origin(it: Item): [number, number] {
  return it.t === 'arrow' ? [0, 0] : [it.x, it.y];
}

export class Stage {
  private live = new Map<string, { g: SVGGElement; html: string }>();

  constructor(private svg: SVGSVGElement) {
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  }

  clear(): void {
    this.svg.innerHTML = '';
    this.live.clear();
  }

  render(items: Item[]): void {
    const keep = new Set<string>();
    for (const it of items) {
      keep.add(it.k);
      const html = inner(it);
      const [x, y] = origin(it);
      const dim = 'dim' in it && it.dim;
      let e = this.live.get(it.k);
      if (!e) {
        const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 's-item s-enter');
        g.style.transform = `translate(${x}px, ${y}px)`;
        g.innerHTML = html;
        this.svg.appendChild(g);
        e = { g, html };
        this.live.set(it.k, e);
        requestAnimationFrame(() => requestAnimationFrame(() => g.classList.remove('s-enter')));
      } else {
        e.g.style.transform = `translate(${x}px, ${y}px)`;
        if (e.html !== html) { e.g.innerHTML = html; e.html = html; }
      }
      e.g.classList.toggle('s-dim', !!dim);
    }
    for (const [k, e] of this.live) {
      if (keep.has(k)) continue;
      this.live.delete(k);
      e.g.classList.add('s-enter');
      setTimeout(() => e.g.remove(), 350);
    }
  }
}
