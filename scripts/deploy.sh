#!/usr/bin/env bash
# Deploy / update ARSE on the VM and verify it.  Run from the repo root:
#
#   scripts/deploy.sh            # pull, build, restart, verify
#   scripts/deploy.sh --no-pull  # deploy the working tree as-is
#
# Steps: certificate preflight → git pull → build → fix data-volume
# ownership (the image runs as uid 1000) → up -d → wait for healthy →
# verify TLS chain + HTTP for every hostname.
set -euo pipefail
cd "$(dirname "$0")/.."

HOSTS=(reference-cleaner.ds-ifs.tuwien.ac.at arse.ds-ifs.tuwien.ac.at)
PULL=1
[[ "${1:-}" == "--no-pull" ]] && PULL=0

red()   { printf '\033[31m%s\033[0m\n' "$*"; }
green() { printf '\033[32m%s\033[0m\n' "$*"; }
FAILED=0
fail()  { red "✗ $*"; FAILED=1; }

[[ -f .env ]] || { red ".env missing — copy .env.example and fill it in"; exit 1; }
# Compose reads TLS_* from .env; mirror its defaults here.
envval() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- | tr -d '"' ; }
TLS_CERT=$(envval TLS_CERT); TLS_CERT=${TLS_CERT:-/etc/ssl/reference-cleaner.pem}
TLS_KEY=$(envval TLS_KEY);   TLS_KEY=${TLS_KEY:-/etc/ssl/key.pem}
ARSE_TLS_CERT=$(envval ARSE_TLS_CERT); ARSE_TLS_CERT=${ARSE_TLS_CERT:-$TLS_CERT}
ARSE_TLS_KEY=$(envval ARSE_TLS_KEY);   ARSE_TLS_KEY=${ARSE_TLS_KEY:-$TLS_KEY}

# ── 1. Certificate preflight ────────────────────────────────────────────
check_cert() {   # host cert key
    local host=$1 cert=$2 key=$3
    if ! sudo test -r "$cert" || ! sudo test -r "$key"; then
        fail "$host: cert/key not found ($cert, $key)"; return
    fi
    local n; n=$(sudo grep -c 'BEGIN CERTIFICATE' "$cert")
    if (( n < 2 )); then
        fail "$host: $cert has no intermediate (leaf only) — run scripts/build_fullchain.sh"
    fi
    if ! sudo openssl x509 -in "$cert" -noout -checkhost "$host" | grep -q "does match"; then
        fail "$host: $cert does not cover this hostname (check its SAN list)"
    fi
    if ! sudo openssl x509 -in "$cert" -noout -checkend $((14*86400)) >/dev/null; then
        fail "$host: $cert expires within 14 days"
    fi
    local kc cc
    kc=$(sudo openssl pkey -in "$key" -pubout 2>/dev/null | sha256sum)
    cc=$(sudo openssl x509 -in "$cert" -noout -pubkey | sha256sum)
    [[ "$kc" == "$cc" ]] || fail "$host: key does not match certificate"
}
echo "== certificate preflight"
check_cert reference-cleaner.ds-ifs.tuwien.ac.at "$TLS_CERT" "$TLS_KEY"
check_cert arse.ds-ifs.tuwien.ac.at "$ARSE_TLS_CERT" "$ARSE_TLS_KEY"
if (( FAILED )); then
    red "Certificate problems above.  nginx would still start, but clients"
    red "will see TLS errors for the affected hostname(s)."
    read -r -p "Continue anyway? [y/N] " ans
    [[ "$ans" == [yY] ]] || exit 1
    FAILED=0
fi

# ── 2. Code + images ────────────────────────────────────────────────────
if (( PULL )); then
    echo "== git pull"
    git pull --ff-only
fi
echo "== build"
docker compose build

# ── 3a. One-off: the data volume was renamed ares-data → arse-data.  Copy
#        the lookup cache and API keys over so nothing is lost; the old
#        volume stays as a backup (remove it with `docker volume rm`).
project=$(docker compose config 2>/dev/null | awk '/^name:/{print $2; exit}')
old_vol="${project}_ares-data"
new_vol="${project}_arse-data"
if docker volume inspect "$old_vol" >/dev/null 2>&1 \
        && ! docker volume inspect "$new_vol" >/dev/null 2>&1; then
    echo "== migrating data volume $old_vol -> $new_vol"
    # The old services no longer exist in the compose file, so stop their
    # containers by label — otherwise the app keeps writing the SQLite
    # cache while it is being copied.
    old=$(docker ps -q --filter "label=com.docker.compose.project=$project" \
              --filter "label=com.docker.compose.service=ares"; \
          docker ps -q --filter "label=com.docker.compose.project=$project" \
              --filter "label=com.docker.compose.service=ares-api")
    [[ -n "$old" ]] && docker stop $old >/dev/null
    docker volume create "$new_vol" >/dev/null
    docker run --rm -v "$old_vol":/from:ro -v "$new_vol":/to alpine \
        sh -c 'cp -a /from/. /to/'
    echo "   done; old volume kept as backup: $old_vol"
fi

# ── 3. Data volume ownership (one-off after the switch to a non-root image)
echo "== data volume ownership"
docker compose run --rm --no-deps --user root --entrypoint sh arse \
    -c 'chown -R 1000:1000 /app/data'

# ── 4. Start + wait for health ──────────────────────────────────────────
echo "== up"
docker compose up -d --remove-orphans
echo "== waiting for services to become healthy (GROBID can take ~4 min)"
deadline=$(( $(date +%s) + 420 ))
while :; do
    states=$(docker compose ps --format '{{.Service}}={{.Health}}')
    if ! grep -qvE '=healthy$' <<<"$states"; then
        green "all healthy"; break
    fi
    if (( $(date +%s) > deadline )); then
        red "timed out waiting:"; echo "$states"; FAILED=1; break
    fi
    sleep 10
done

# ── 5. Verify from the outside ──────────────────────────────────────────
echo "== verify"
for host in "${HOSTS[@]}"; do
    # Resolve to this machine so the check doesn't depend on hairpin NAT.
    verify=$(echo | openssl s_client -connect 127.0.0.1:443 -servername "$host" \
             -verify_hostname "$host" 2>/dev/null | grep -E 'Verify return code' || true)
    if grep -q 'code: 0 (ok)' <<<"$verify"; then
        green "✓ $host: TLS chain verifies"
    else
        fail "$host: TLS — ${verify:-no handshake}"
    fi
    code=$(curl -s -o /dev/null -w '%{http_code}' --resolve "$host:443:127.0.0.1" "https://$host/" || true)
    [[ "$code" == 200 ]] && green "✓ $host: UI responds (200)" || fail "$host: UI returned $code"
    code=$(curl -s -o /dev/null -w '%{http_code}' --resolve "$host:443:127.0.0.1" "https://$host/api-health" || true)
    [[ "$code" == 200 ]] && green "✓ $host: API health (200)" || fail "$host: /api-health returned $code"
    # The page itself loads even when Streamlit's WebSocket is refused
    # (it then spins forever), so check the WebSocket handshake as a
    # browser on this hostname would send it.  -k: TLS is checked above.
    code=$(curl -sk --http1.1 -o /dev/null -w '%{http_code}' --max-time 5 \
             -H 'Connection: Upgrade' -H 'Upgrade: websocket' \
             -H 'Sec-WebSocket-Version: 13' -H 'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==' \
             -H "Origin: https://$host" \
             --resolve "$host:443:127.0.0.1" "https://$host/_stcore/stream" || true)
    [[ "$code" == 101 ]] && green "✓ $host: app WebSocket (101)" \
        || fail "$host: WebSocket /_stcore/stream returned $code (page would never finish loading)"
    # A first page load fetches ~80 static chunks at once over HTTP/2;
    # nginx's per-IP limits must let that burst through.
    asset=$(curl -sk --resolve "$host:443:127.0.0.1" "https://$host/" \
            | grep -o '/static/js/[^"]*\.js' | head -1)
    if [[ -n "$asset" ]]; then
        args=(); for i in $(seq 80); do args+=(-o /dev/null "https://$host$asset?n=$i"); done
        ok=$(curl -sk --http2 -Z --parallel-max 100 --resolve "$host:443:127.0.0.1" \
                 -w '%{http_code}\n' "${args[@]}" | grep -c '^200$' || true)
        [[ "$ok" == 80 ]] && green "✓ $host: 80 parallel asset requests (200)" \
            || fail "$host: only $ok of 80 parallel asset requests succeeded (rate/connection limit?)"
    fi
done

if (( FAILED )); then
    red "Deploy finished with problems (see above).  Logs: docker compose logs --tail 100"
    exit 1
fi
green "Deploy OK."
echo "If this deploy includes the cache-poisoning fix, purge stale negatives once:"
echo "  docker compose exec arse python scripts/purge_negative_cache.py"
