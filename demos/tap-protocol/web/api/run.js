// POST /api/run  — Vercel Node runtime
//
// Proxies {workload, params?} to the gateway's POST /run (async job: the
// gateway replies 202 {id} immediately; progress flows via /api/events and
// the graph via /api/graph?id=N). GATEWAY_URL is held in env; never sent
// to the browser.

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).json({ error: 'method not allowed' });
    return;
  }

  const target = process.env.GATEWAY_URL;
  if (!target) {
    res.status(503).json({ error: 'GATEWAY_URL not configured (cluster is offline)' });
    return;
  }

  const body = (req.body && typeof req.body === 'object') ? req.body : (() => {
    try { return JSON.parse(req.body || '{}'); } catch { return null; }
  })();
  if (!body || typeof body.workload !== 'string') {
    res.status(400).json({ error: 'workload required' });
    return;
  }

  try {
    const upstream = await fetch(`${target.replace(/\/$/, '')}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workload: body.workload, params: body.params || {} }),
      signal: AbortSignal.timeout(30000),
    });
    const data = await upstream.json().catch(() => ({}));
    res.status(upstream.status).json(data);
  } catch (e) {
    res.status(502).json({ error: `gateway unreachable: ${e.message}` });
  }
};
