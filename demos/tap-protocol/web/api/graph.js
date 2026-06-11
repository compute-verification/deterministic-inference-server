// GET /api/graph?id=N  — Vercel Node runtime
//
// Proxies the gateway's GET /run/<id>/graph: the task graph generated from
// the live run's capture, in the same {scene: graph} shape as the demo's
// protocol-graphs.json, so the embedded viz loads either via ?src=.

module.exports = async function handler(req, res) {
  const target = process.env.GATEWAY_URL;
  if (!target) {
    res.status(503).json({ error: 'GATEWAY_URL not configured (cluster is offline)' });
    return;
  }

  const id = parseInt((req.query && req.query.id) || '', 10);
  if (!Number.isFinite(id) || id <= 0) {
    res.status(400).json({ error: 'id required' });
    return;
  }

  try {
    const upstream = await fetch(
      `${target.replace(/\/$/, '')}/run/${id}/graph`,
      { signal: AbortSignal.timeout(60000) });
    const data = await upstream.json().catch(() => ({}));
    res.status(upstream.status).json(data);
  } catch (e) {
    res.status(502).json({ error: `gateway unreachable: ${e.message}` });
  }
};
