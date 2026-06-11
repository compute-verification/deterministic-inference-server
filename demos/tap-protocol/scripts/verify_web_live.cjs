// Headless check of LIVE mode: page -> /api/run -> 4-node stack -> SSE events
// -> verdict -> /api/graph. Needs dev_server.cjs (or the Vercel deploy) with
// a reachable gateway behind it.
//   BASE_URL=http://127.0.0.1:8766 node demos/tap-protocol/scripts/verify_web_live.cjs
const path = require("path");
const { createRequire } = require("module");
const vizRequire = createRequire(
  path.join(__dirname, "..", "..", "proof-compare", "viz", "package.json"));
const { chromium } = vizRequire("playwright-core");

const BASE = process.env.BASE_URL || "http://127.0.0.1:8766";
const EXE = process.env.CHROME ||
  "/home/jon/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome";

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage();
  const errors = [];
  // favicon 404s on the dev server are noise, not failures
  const IGNORABLE = /Failed to load resource.*404/;
  page.on("console", (m) => {
    if (m.type() === "error" && !IGNORABLE.test(m.text())) errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));

  await page.goto(BASE + "/index.html", { waitUntil: "networkidle" });

  // switch to live; the probe must come back online
  await page.locator("#mode-live").click();
  await page.waitForFunction(
    () => document.getElementById("status").textContent === "live",
    null, { timeout: 20000 });
  console.log("live probe: online");

  // run a real (mock-stack) job through the cluster
  await page.locator('.scenario[data-wl="spec"] button').click();
  await page.waitForSelector("tr .mode-tag", { timeout: 20000 });
  console.log("run row:", (await page.locator(".t-scenario .mode-tag").first().textContent()).trim());

  await page.waitForSelector(".badge.ok", { timeout: 120000 });
  console.log("live verdict: Verified");

  // graph panel must load the run's graph from /api/graph
  await page.waitForSelector("#graph-card.visible", { timeout: 20000 });
  const src = await page.locator("#graph-frame").getAttribute("src");
  if (!/api%2Fgraph|api\/graph|graph%3Fid|graph\?id/.test(decodeURIComponent(src)))
    throw new Error("graph iframe src is not the live graph endpoint: " + src);
  const fl = page.frameLocator("#graph-frame");
  await fl.locator(".tab.active").waitFor({ timeout: 30000 });
  console.log("live graph iframe loaded:", decodeURIComponent(src));
  const active = await fl.locator(".tab.active").textContent();
  if (!/Speculative/.test(active)) throw new Error("wrong active scene: " + active);

  console.log("console errors:", errors.length ? errors : "none");
  await browser.close();
  if (errors.length) process.exit(2);
  console.log("PASS");
})();
