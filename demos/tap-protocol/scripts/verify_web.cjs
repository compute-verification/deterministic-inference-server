// Headless check of the 4-scenario demo page (demo mode + offline live probe).
//
// playwright-core comes from the viz package's dev deps (pnpm install there
// first). Run from anywhere:
//   BASE_URL=http://127.0.0.1:8765 node demos/tap-protocol/scripts/verify_web.cjs
// or against the live deployment:
//   BASE_URL=https://<deploy>.vercel.app node .../verify_web.cjs
const path = require("path");
const { createRequire } = require("module");
const vizRequire = createRequire(
  path.join(__dirname, "..", "..", "proof-compare", "viz", "package.json"));
const { chromium } = vizRequire("playwright-core");

const BASE = process.env.BASE_URL || "http://127.0.0.1:8765";
const EXE = process.env.CHROME ||
  "/home/jon/.cache/ms-playwright/chromium-1217/chrome-linux64/chrome";
// On a bare static server /api/* 404s (no serverless fns) — that's the
// graceful-offline path, not an error. Same for a missing favicon.
const IGNORABLE = /Failed to load resource.*(404|503)/;

(async () => {
  const browser = await chromium.launch({ executablePath: EXE });
  const page = await browser.newPage();
  const errors = [];
  page.on("console", (m) => {
    if (m.type() === "error" && !IGNORABLE.test(m.text())) errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));

  await page.goto(BASE + "/index.html", { waitUntil: "networkidle" });

  // 1. four scenario cards with stats from the manifest
  const cards = await page.locator(".scenario").count();
  if (cards !== 4) throw new Error(`expected 4 cards, got ${cards}`);
  console.log("cards:", cards);
  console.log("mode note:", (await page.locator("#mode-note").textContent()).trim());
  console.log("verified chips:", await page.locator(".verdict-chip.ok").count());

  // 2. replay the inference scenario to the verdict
  await page.locator('.scenario[data-wl="inference"] button').click();
  await page.waitForSelector(".badge.ok", { timeout: 60000 });
  console.log("inference replay: Verified");
  console.log("digest row:",
    (await page.locator(".digest-row .pair").first().textContent()).trim().slice(0, 80));

  // 3. graph panel appears and the embedded viz renders (frameLocator waits
  // for the iframe to attach + navigate — page.frame() raced on slow nets)
  await page.waitForSelector("#graph-card.visible", { timeout: 15000 });
  const fl = page.frameLocator("#graph-frame");
  await fl.locator(".tab").first().waitFor({ timeout: 30000 });
  console.log("iframe tabs:", (await fl.locator(".tab").allTextContents()).join("|"),
    "active:", await fl.locator(".tab.active").textContent());

  // 4. replay the coding scenario too; meta must switch to the coding stats
  const metaBefore = await page.locator("#graph-meta").textContent();
  await page.locator('.scenario[data-wl="coding"] button').click();
  await page.waitForFunction(
    (prev) => {
      const t = document.getElementById("graph-meta").textContent;
      return t && t !== prev;
    }, metaBefore, { timeout: 120000 });
  console.log("coding replay graph meta:", (await page.locator("#graph-meta").textContent()).trim());

  // 5. live toggle: lands "offline" on a static deploy (graceful message,
  // buttons disabled) or "live" when a gateway is behind the deployment
  await page.locator("#mode-live").click();
  await page.waitForFunction(
    () => ["offline", "live"].includes(document.getElementById("status").textContent),
    null, { timeout: 15000 });
  const probe = await page.locator("#status").textContent();
  const disabled = await page.locator('.scenario[data-wl="inference"] button').isDisabled();
  console.log(`live probe: ${probe}; buttons disabled: ${disabled}`);
  if (probe === "offline" && !disabled) throw new Error("offline must disable run buttons");
  if (probe === "live" && disabled) throw new Error("live must enable run buttons");

  await page.locator("#mode-demo").click();
  const btn = await page.locator('.scenario[data-wl="spec"] button').textContent();
  if (!/Replay/.test(btn)) throw new Error("demo mode button text wrong: " + btn);

  console.log("console errors:", errors.length ? errors : "none");
  await browser.close();
  if (errors.length) process.exit(2);
  console.log("PASS");
})();
