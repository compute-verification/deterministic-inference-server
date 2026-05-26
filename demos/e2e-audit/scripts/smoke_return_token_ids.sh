#!/usr/bin/env bash
# Verify that the locally-running vLLM honours `return_token_ids: true`.
# Usage: ./smoke_return_token_ids.sh <vllm_port>
set -euo pipefail
PORT="${1:-8001}"
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

echo "[smoke] POST /v1/chat/completions with return_token_ids=true"
RESP=$(curl -sS -H 'Content-Type: application/json' \
  "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -d "$(python3 -c "import json,sys; print(json.dumps({
    'model': '${MODEL}',
    'messages': [{'role': 'user', 'content': 'Say exactly HELLO'}],
    'max_tokens': 4,
    'temperature': 0,
    'seed': 42,
    'return_token_ids': True,
  }))")")
echo "$RESP" | python3 -m json.tool | head -40
echo "---"
echo "[smoke] checking choices[0].token_ids is present and non-empty"
python3 -c "
import sys, json
r = json.loads('''$RESP''')
choice = r['choices'][0]
tids = choice.get('token_ids')
if not tids or not isinstance(tids, list):
    print('FAIL: choices[0].token_ids missing or not a list; keys=', list(choice.keys()))
    sys.exit(1)
print('OK: got', len(tids), 'token_ids:', tids)
"
