// Render the HTML/CSS panels into GitHub-compatible images.
// Install the rendering tool separately from the pipeline:
// npm install --prefix /tmp/kbot-doc-tools playwright@1.65.1
// /tmp/kbot-doc-tools/node_modules/.bin/playwright install chromium
// KBOT_PLAYWRIGHT_MODULE=/path/to/node_modules/playwright node docs/visuals/render.cjs
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { chromium } = require(process.env.KBOT_PLAYWRIGHT_MODULE || 'playwright');
const root = __dirname;
const targets = {
  architecture: 'architecture.png',
  components: 'components.png',
  topology: 'topology.png',
  'json-output': 'json-output.png',
  analysis: 'analysis.png',
};
const mime = { '.html': 'text/html', '.css': 'text/css', '.json': 'application/json', '.svg': 'image/svg+xml', '.png': 'image/png' };
const server = http.createServer((req, res) => {
  const name = decodeURIComponent(new URL(req.url, 'http://localhost').pathname);
  const file = path.resolve(root, '.' + (name === '/' ? '/index.html' : name));
  if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    res.writeHead(404); res.end(); return;
  }
  res.writeHead(200, { 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
  fs.createReadStream(file).pipe(res);
});
(async () => {
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1800, height: 1200 }, deviceScaleFactor: 1.5 });
    const errors = []; page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:' + server.address().port, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.documentElement.dataset.ready === 'true', null, { timeout: 15000 });
    await page.evaluate(async () => {
      await document.fonts.ready;
      await Promise.all([...document.images].map(image => image.decode()));
    });
    if (errors.length) throw new Error(errors.join('\n'));
    for (const [id, filename] of Object.entries(targets)) {
      await page.locator('#' + id).screenshot({ path: path.join(root, filename) });
      console.log(filename);
    }
  } finally { await browser.close(); server.close(); }
})().catch(error => { console.error(error); server.close(); process.exitCode = 1; });
