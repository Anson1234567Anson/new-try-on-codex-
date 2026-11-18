// Opens a lesson in a real browser, steps through it, and saves a screenshot of every step.
// Usage: node scripts/walk.cjs <base-url> <lesson-id> <out-dir> [en|zh] [wait-ms]
// e.g.   node scripts/walk.cjs http://localhost:5174/ embeddings /tmp/shots zh
const { chromium } = require('playwright');
const fs = require('fs');
(async () => {
  const [base, id, out, lang = 'en', wait = '900'] = process.argv.slice(2);
  fs.mkdirSync(out, { recursive: true });
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  p.on('pageerror', (e) => errors.push('page error: ' + e.message));
  p.on('console', (m) => { if (m.type() === 'error') errors.push('console: ' + m.text()); });
  await p.goto(base);
  await p.evaluate((l) => localStorage.setItem('tfl:lang', JSON.stringify(l)), lang);
  await p.goto(`${base}#/${id}/1`);
  await p.reload();
  await p.waitForTimeout(800);
  const n = await p.evaluate(() => document.querySelectorAll('.dots li').length);
  if (!n) { console.log('lesson not found: is it exported and is its id in MODULES?'); }
  for (let s = 1; s <= n; s++) {
    await p.goto(`${base}#/${id}/${s}`);
    await p.waitForTimeout(+wait);
    const file = `${out}/${id}-${lang}-${s}.png`;
    await p.screenshot({ path: file });
    const metrics = await p.evaluate(() => document.getElementById('metrics').innerText.replace(/\n/g, ' '));
    // text drawn outside the 900×520 diagram
    const outside = await p.evaluate(() => {
      const svg = document.getElementById('svg'); const box = svg.getBoundingClientRect();
      return [...svg.querySelectorAll('text, rect')].filter((e) => {
        const r = e.getBoundingClientRect();
        return r.width > 0 && (r.left < box.left - 1 || r.right > box.right + 1 || r.top < box.top - 1 || r.bottom > box.bottom + 1);
      }).map((e) => e.textContent || e.tagName).slice(0, 5);
    });
    console.log(`step ${s}: ${file} | ${metrics}${outside.length ? ' | OUTSIDE DIAGRAM: ' + outside.join(', ') : ''}`);
  }
  console.log(errors.length ? errors.join('\n') : 'no browser errors');
  await b.close();
})();
