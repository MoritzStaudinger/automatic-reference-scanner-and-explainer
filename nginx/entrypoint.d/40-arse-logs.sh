#!/bin/sh
# Runs as root before nginx starts (official image: /docker-entrypoint.d/).
# The access-log path contains a variable, so worker processes (user nginx)
# open the file themselves and need write access to the directory.
set -e
dir=/var/log/nginx/arse
mkdir -p "$dir"
chown nginx:nginx "$dir"
chmod 750 "$dir"
# Retention: ~13 months.
find "$dir" -name 'access-*.log*' -mtime +400 -delete
