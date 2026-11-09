# Shared by every HTTPS server block in default.conf.
# (.inc, not .conf, so nginx's conf.d/*.conf glob doesn't load it twice.)

ssl_protocols TLSv1.2 TLSv1.3;
ssl_prefer_server_ciphers off;
ssl_session_cache shared:ares_tls:10m;
ssl_session_timeout 1d;

add_header Strict-Transport-Security "max-age=31536000" always;
add_header X-Content-Type-Options "nosniff" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
# Streamlit needs inline scripts/styles, so only framing is restricted;
# its own components are same-origin iframes.
add_header Content-Security-Policy "frame-ancestors 'self'" always;
add_header X-Frame-Options "SAMEORIGIN" always;

# Streamlit enforces its own 25 MB upload limit (config.toml) — keep
# nginx a bit above so users see Streamlit's message, not a bare 413.
client_max_body_size 30m;
# Per-IP concurrent connections.  Over HTTP/2 nginx counts every stream,
# and Streamlit's first page load fetches ~80 JS/CSS chunks at once —
# a limit of 30 answered most of them with 429 and the page never loaded.
# The API keeps a tight limit (below); static assets have their own.
limit_conn ares_conn 100;

proxy_http_version 1.1;
proxy_set_header Host              $host;
proxy_set_header X-Real-IP         $remote_addr;
proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
proxy_set_header X-Forwarded-Proto $scheme;

# HTTP API.  Longest-prefix match wins over "/".
set $arse_ui  arse:8501;      # variables → re-resolved via `resolver`
set $arse_api arse-api:8500;

location /api/ {
    limit_req zone=ares_api burst=20 nodelay;
    limit_conn ares_conn 30;
    proxy_pass http://$arse_api;
    # Sync /api/v1/process can take minutes for an uncached paper.
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
    # Stream /results.jsonl through without buffering the whole batch.
    proxy_buffering off;
}

location = /api/v1/batch {
    limit_req zone=ares_api burst=5 nodelay;
    limit_conn ares_conn 30;
    client_max_body_size 200m;
    proxy_pass http://$arse_api;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
    proxy_request_buffering off;
}

location = /api-health {
    proxy_pass http://$arse_api/healthz;
}

# Streamlit's static assets: content-hashed file names, so they can be
# cached by the browser and need no request-rate limit.  (`expires`, not
# add_header — an add_header here would drop the security headers above.)
location /static/ {
    limit_conn ares_conn 300;
    proxy_pass http://$arse_ui;
    expires 7d;
}

# Streamlit UI.
location / {
    limit_req zone=ares_ui burst=100 nodelay;
    proxy_pass http://$arse_ui;
    # Streamlit's live UI runs over a WebSocket (/_stcore/stream).
    proxy_set_header Upgrade    $http_upgrade;
    proxy_set_header Connection $connection_upgrade;
    # Any proxy_set_header here drops ALL inherited ones, so the headers
    # above (server level) must be repeated.  Without Host, Streamlit saw
    # "arse:8501", which doesn't match the browser's Origin, and rejected
    # every WebSocket with 403 — the page never loaded.  (Locally the
    # origin was localhost, which Streamlit always allows.)
    proxy_set_header Host              $host;
    proxy_set_header X-Real-IP         $remote_addr;
    proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    # Idle WebSocket timeout.  Long lookups keep sending progress, but a
    # quiet tab shouldn't be dropped after 60 s.
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
}
