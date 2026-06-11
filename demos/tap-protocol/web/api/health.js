// GET /api/health  — Vercel Node runtime
//
// Probes the gateway so the frontend's Live toggle can tell "cluster online"
// from "nothing deployed behind this page" without exposing GATEWAY_URL.

module.exports = async function handler(req, res) {
  const target = process.env.GATEWAY_URL;
  if (!target) {
    res.status(503).json({ ok: false, reason: 'GATEWAY_URL not configured' });
    return;
  }
  try {
    const upstream = await fetch(`${target.replace(/\/$/, '')}/health`,
      { signal: AbortSignal.timeout(8000) });
    if (upstream.ok) res.status(200).json({ ok: true });
    else res.status(503).json({ ok: false, reason: `gateway HTTP ${upstream.status}` });
  } catch (e) {
    res.status(503).json({ ok: false, reason: `gateway unreachable: ${e.message}` });
  }
};
