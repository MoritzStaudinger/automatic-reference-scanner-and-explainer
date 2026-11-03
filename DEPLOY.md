# Deploying ARSE

Production runs on one VM with Docker Compose: `grobid`, `arse` (Streamlit
UI), `arse-api` (FastAPI) and `nginx` (TLS, the only service on the host
network). It serves two hostnames:

- `reference-cleaner.ds-ifs.tuwien.ac.at`
- `arse.ds-ifs.tuwien.ac.at`

Both DNS names resolve to the VM (128.131.169.216).

## Requirements

- Docker with Compose v2.24+
- About 12 GB RAM for the containers: GROBID needs up to 6 GB, and arse and
  arse-api need up to 4 GB each under load. GROBID is the one that matters;
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

The services, containers and data volume were renamed from `ares*` to
`arse*`. On the first deploy after that, `deploy.sh` copies the old
`<project>_ares-data` volume (lookup cache, API keys) into the new
`<project>_arse-data` and keeps the old one as a backup. Once everything
works, remove it with `docker volume rm <project>_ares-data`.

The environment variables were renamed as well (`ARES_API_KEY` →
`ARSE_API_KEY`, `ARES_CONTACT_EMAIL` → `ARSE_CONTACT_EMAIL`, …; see
`.env.example`). An existing `.env` with the old `ARES_*` names keeps
working (`arse_env.py` maps them); rename the lines when convenient.

The first deploy after the cache-poisoning fix needs one extra command:

```bash
docker compose exec arse python scripts/purge_negative_cache.py
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

The HTTP API refuses all requests (403) until at least one key exists.
`/healthz` is the only exception.

```bash
docker compose exec arse-api python scripts/manage_keys.py add alice --expires 2027-06-30
docker compose exec arse-api python scripts/manage_keys.py list
docker compose exec arse-api python scripts/manage_keys.py revoke alice
```

Keys are stored in `/app/data/api_keys.json` on the shared volume. Changes
take effect without a restart.

## Limits for the public instance

| What | Where | Value |
|---|---|---|
| Upload size | `.streamlit/config.toml` / nginx | 25 MB / 30 MB (batch API 200 MB) |
| Request rate per IP | `nginx/conf.d/default.conf` | UI 20 r/s (burst 100), API 2 r/s (burst 20) |
| Connections per IP | nginx | 30 |
| Concurrent synchronous API runs | `ARSE_API_SYNC_CONCURRENCY` | 2 |
| Concurrent GROBID full-document parses | `ARSE_GROBID_FULLTEXT_CONCURRENCY` | 1 per process |

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
- `docker inspect arse-grobid --format '{{.State.OOMKilled}}'` returns
  `true` when GROBID ran out of memory. Raise its `mem_limit`, or lower
  `ARSE_GROBID_FULLTEXT_CONCURRENCY`.
- When GROBID is down, the UI falls back to the pymupdf4llm parser and
  shows a warning. The API reports `summary.parser_fallback`.
