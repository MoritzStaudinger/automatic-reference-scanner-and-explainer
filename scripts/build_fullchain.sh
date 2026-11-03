#!/usr/bin/env bash
# Build a full-chain PEM (leaf + intermediate) from a leaf certificate.
#
# nginx must send the intermediate itself: without it browsers that don't
# fetch missing intermediates, curl, and every API client fail with
# "unable to verify the first certificate".  TU Wien/GEANT certificates
# carry the intermediate's download URL (AIA caIssuers); this script fetches
# it, converts it to PEM, verifies the chain and writes the result.
#
#   scripts/build_fullchain.sh /etc/ssl/reference-cleaner.pem /etc/ssl/reference-cleaner.fullchain.pem
#
# Then point TLS_CERT (or ARSE_TLS_CERT) in .env at the output file.
set -euo pipefail

leaf=${1:?usage: build_fullchain.sh <leaf.pem> <out-fullchain.pem>}
out=${2:?usage: build_fullchain.sh <leaf.pem> <out-fullchain.pem>}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# Only the first certificate of the input counts as the leaf.
openssl x509 -in "$leaf" -out "$tmp/leaf.pem"
url=$(openssl x509 -in "$tmp/leaf.pem" -noout -ext authorityInfoAccess 2>/dev/null \
      | sed -n 's/.*CA Issuers - URI:\(.*\)/\1/p' | head -1)
if [[ -z "$url" ]]; then
    echo "error: leaf has no CA Issuers URL; download the intermediate manually" >&2
    exit 1
fi
echo "intermediate: $url"
curl -fsSL "$url" -o "$tmp/issuer.bin"
if grep -q "BEGIN CERTIFICATE" "$tmp/issuer.bin"; then
    cp "$tmp/issuer.bin" "$tmp/issuer.pem"
else
    openssl x509 -inform DER -in "$tmp/issuer.bin" -out "$tmp/issuer.pem"
fi

# The intermediate must verify against the system trust store, and the
# leaf against the intermediate.
openssl verify -untrusted "$tmp/issuer.pem" "$tmp/leaf.pem"
cat "$tmp/leaf.pem" "$tmp/issuer.pem" > "$out"
chmod 644 "$out"
echo "wrote $out ($(grep -c 'BEGIN CERTIFICATE' "$out") certificates)"
openssl x509 -in "$out" -noout -subject -enddate -ext subjectAltName
