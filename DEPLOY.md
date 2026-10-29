# Deploying ARES

Production runs on one VM with Docker Compose: `grobid`, `ares` (Streamlit
UI), `ares-api` (FastAPI) and `nginx` (TLS, the only service on the host
network). It serves two hostnames:

- `reference-cleaner.ds-ifs.tuwien.ac.at`
- `arse.ds-ifs.tuwien.ac.at`

Both DNS names resolve to the VM (128.131.169.216).

## Requirements

- Docker with Compose v2.24+
- About 12 GB RAM for the containers: GROBID needs up to 6 GB, and ares and
  ares-api need up to 4 GB each under load. GROBID is the one that matters;
  with less memory it is OOM-killed while parsing.
- Full-chain TLS certificates for both hostnames. See TLS below.

## Routine deploy

```bash
cd /path/to/ReferenceCleaner
scripts/deploy.sh
```

The script runs these steps in order:

1. Checks the certificates: full chain present, hostname covered, key
   matches, not expiring within 14 days.
2. Runs `git pull --ff-only`.
3. Builds the image.
4. Fixes the data volume's ownership. The image runs as uid 1000.
5. Starts the stack and waits until every container is healthy.
6. Checks each hostname from the VM: the TLS chain verifies, the UI returns
   200, and `/api-health` returns 200.

The script exits non-zero if any check fails.

The first deploy after the cache-poisoning fix needs one extra command:

```bash
docker compose exec ares python scripts/purge_negative_cache.py
```

## TLS

nginx has to send the **full chain**, meaning the leaf certificate plus the
GEANT/HARICA intermediate. With only the leaf, Chrome still works, but curl,
API clients and some browsers fail with "unable to verify the first
certificate". To build the chain from the leaf TU Wien issued:

```bash
sudo scripts/build_fullchain.sh /etc/ssl/reference-cleaner.pem /etc/ssl/reference-cleaner.fullchain.pem
```

Then set the paths in `.env`:

```
TLS_CERT=/etc/ssl/reference-cleaner.fullchain.pem
TLS_KEY=/etc/ssl/key.pem
# Only needed if arse gets its own certificate:
ARSE_TLS_CERT=
ARSE_TLS_KEY=
```

nginx chooses the certificate by hostname (SNI). Left empty, `ARSE_TLS_*`
reuse `TLS_*`. That works once the certificate's SAN list contains **both**
names, which is the simplest setup: request one certificate from TU Wien
with both SANs.

**Renewal:** replace the files, rebuild the chain, then run
`docker compose restart nginx`.

**Check from outside:**

```bash
for h in reference-cleaner.ds-ifs.tuwien.ac.at arse.ds-ifs.tuwien.ac.at; do
  echo | openssl s_client -connect $h:443 -servername $h -verify_hostname $h 2>/dev/null | grep 'Verify return'
done
```

## API keys

The HTTP API refuses all requests (503) until at least one key exists.
`/healthz` is the only exception.

```bash
docker compose exec ares-api python scripts/manage_keys.py add alice --expires 2027-06-30
docker compose exec ares-api python scripts/manage_keys.py list
docker compose exec ares-api python scripts/manage_keys.py revoke alice
```

Keys are stored in `/app/data/api_keys.json` on the shared volume. Changes
take effect without a restart.

## Limits for the public instance

| What | Where | Value |
|---|---|---|
| Upload size | `.streamlit/config.toml` / nginx | 25 MB / 30 MB (batch API 200 MB) |
| Request rate per IP | `nginx/conf.d/default.conf` | UI 20 r/s (burst 100), API 2 r/s (burst 20) |
| Connections per IP | nginx | 30 |
| Concurrent synchronous API runs | `ARES_API_SYNC_CONCURRENCY` | 2 |
| Concurrent GROBID full-document parses | `ARES_GROBID_FULLTEXT_CONCURRENCY` | 1 per process |

## Local stack (same proxy path as production)

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 30 \
  -subj /CN=reference-cleaner.ds-ifs.tuwien.ac.at -keyout dev.key -out dev.pem
TLS_CERT=$PWD/dev.pem TLS_KEY=$PWD/dev.key \
  docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

| Service | URL |
|---|---|
| UI via nginx | https://localhost:8443 |
| UI directly | http://localhost:8501 |
| API | http://localhost:8500 (no key needed in dev) |
| GROBID | http://localhost:8070 |

## Troubleshooting

- `docker compose ps` shows the health of every service.
- `docker compose logs --tail 100 <service>` shows recent logs.
- `docker inspect ares-grobid --format '{{.State.OOMKilled}}'` returns
  `true` when GROBID ran out of memory. Raise its `mem_limit`, or lower
  `ARES_GROBID_FULLTEXT_CONCURRENCY`.
- When GROBID is down, the UI falls back to the pymupdf4llm parser and
  shows a warning. The API reports `summary.parser_fallback`.
