// Local stand-in for the Vercel deployment: serves web/ statically and
// emulates the /api/* serverless functions against a gateway (mock stack or
// the real H100 box). Lets the Live mode path be tested without deploying.
//
//   GATEWAY_URL=http://127.0.0.1:28300 node scripts/dev_server.cjs [port]
const http = require("http");
const fs = require("fs");
const path = require("path");

const WEB = path.join(__dirname, "..", "web");
const GATEWAY = (process.env.GATEWAY_URL || "").replace(/\/$/, "");
const PORT = Number(process.argv[2] || 8766);

const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".cjs": "text/javascript",
  ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml",
  ".png": "image/png", ".ico": "image/x-icon", ".map": "application/json",
};

function sendJSON(res, code, obj) {
  const b = JSON.stringify(obj);
  res.writeHead(code, { "Content-Type": "application/json" });
  res.end(b);
}

async function proxyJSON(res, url, init) {
  if (!GATEWAY) return sendJSON(res, 503, { error: "GATEWAY_URL not configured" });
  try {
    const up = await fetch(url, init);
    const data = await up.json().catch(() => ({}));
    sendJSON(res, up.status, data);
  } catch (e) {
    sendJSON(res, 502, { error: `gateway unreachable: ${e.message}` });
  }
}

http.createServer(async (req, res) => {
  const u = new URL(req.url, "http://x");

  if (u.pathname === "/api/health")
    return proxyJSON(res, `${GATEWAY}/health`);
  if (u.pathname === "/api/run" && req.method === "POST") {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () =>
      proxyJSON(res, `${GATEWAY}/run`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body,
      }));
    return;
  }
  if (u.pathname === "/api/graph")
    return proxyJSON(res, `${GATEWAY}/run/${u.searchParams.get("id")}/graph`);
  if (u.pathname === "/api/events") {
    if (!GATEWAY) return sendJSON(res, 503, { error: "GATEWAY_URL not configured" });
    try {
      const up = await fetch(`${GATEWAY}/events`,
        { headers: { Accept: "text/event-stream" } });
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      });
      const reader = up.body.getReader();
      req.on("close", () => reader.cancel().catch(() => {}));
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        res.write(Buffer.from(value));
      }
      res.end();
    } catch (e) {
      sendJSON(res, 502, { error: `gateway unreachable: ${e.message}` });
    }
    return;
  }

  // static files from web/
  let p = path.normalize(path.join(WEB, u.pathname === "/" ? "index.html" : u.pathname));
  if (!p.startsWith(WEB)) { res.writeHead(403); return res.end(); }
  if (fs.existsSync(p) && fs.statSync(p).isDirectory()) p = path.join(p, "index.html");
  fs.readFile(p, (err, data) => {
    if (err) { res.writeHead(404); return res.end("not found"); }
    res.writeHead(200, { "Content-Type": MIME[path.extname(p)] || "application/octet-stream" });
    res.end(data);
  });
}).listen(PORT, () =>
  console.log(`[dev_server] http://127.0.0.1:${PORT} -> web/, gateway=${GATEWAY || "<none>"}`));
