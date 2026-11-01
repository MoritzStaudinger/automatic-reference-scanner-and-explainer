"""
Mint, list, and revoke API keys for the ARSE HTTP API.

The keys live in ``data/api_keys.json`` (override with the
``ARES_API_KEYS_FILE`` env var).  The running API watches the file's
mtime and reloads on every request, so changes apply without a restart.

Usage
-----
::

    # Mint a new key for 'alice', print it (HAND IT OVER ONCE — we store
    # the value but don't print it again later)
    python scripts/manage_keys.py add alice

    # Mint a key that auto-expires
    python scripts/manage_keys.py add bob --expires 2026-12-31

    # See who has access
    python scripts/manage_keys.py list

    # Revoke by label (preferred) or by key value
    python scripts/manage_keys.py revoke alice
    python scripts/manage_keys.py revoke ares_abc123...

In Docker
---------
::

    docker compose exec arse-api \\
        python /app/scripts/manage_keys.py add alice
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import secrets
import sys
from pathlib import Path


DEFAULT_PATH = Path(
    os.getenv("ARES_API_KEYS_FILE",
              Path(__file__).resolve().parent.parent / "data" / "api_keys.json")
)


def _load(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text() or "{}")
    except json.JSONDecodeError as exc:
        sys.exit(f"error: {path} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        sys.exit(f"error: {path} root must be a JSON object")
    return data


def _save(path: Path, data: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a tmp + rename so a crash mid-write can't leave a
    # truncated keys file that locks everyone out.
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    try:
        os.chmod(path, 0o600)        # secrets — owner-only
    except OSError:
        pass


def _new_key() -> str:
    # 32 url-safe bytes ≈ 256 bits of entropy.  `ares_` prefix makes
    # it easy to recognise (and to grep for in leaked logs).
    return "ares_" + secrets.token_urlsafe(32)


def _find_by_label(keys: dict[str, dict], label: str) -> str | None:
    for k, meta in keys.items():
        if (meta or {}).get("label") == label:
            return k
    return None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_add(args: argparse.Namespace) -> int:
    keys = _load(args.file)
    if _find_by_label(keys, args.label):
        sys.exit(f"error: a key with label {args.label!r} already exists "
                 f"(revoke it first)")
    if args.expires:
        try:
            exp = datetime.datetime.fromisoformat(args.expires.replace("Z", "+00:00"))
        except ValueError:
            sys.exit(f"error: --expires {args.expires!r} is not ISO-8601 "
                     f"(YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)")
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=datetime.timezone.utc)
        args.expires = exp.isoformat()
    new = _new_key()
    keys[new] = {
        "label":       args.label,
        "created_at":  datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat(),
        "expires_at":  args.expires,
    }
    _save(args.file, keys)
    print(f"Minted key for {args.label!r}.  Hand this out — it won't be "
          f"shown again:\n\n  {new}\n")
    if args.expires:
        print(f"  (expires {args.expires})")
    print(f"  Keys file: {args.file}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    keys = _load(args.file)
    if not keys:
        print(f"(no keys in {args.file})")
        return 0
    print(f"{'LABEL':<24}  {'CREATED':<19}  {'EXPIRES':<19}  KEY (first 12 chars)")
    print("-" * 90)
    for k, meta in sorted(keys.items(), key=lambda kv: (kv[1] or {}).get("label", "")):
        meta = meta or {}
        print(f"{(meta.get('label') or '?'):<24}  "
              f"{(meta.get('created_at') or '?'):<19}  "
              f"{(meta.get('expires_at') or '-'):<19}  "
              f"{k[:12]}…")
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    keys = _load(args.file)
    target = args.target

    # Resolve either by label or by raw key value
    if target in keys:
        key_value = target
    else:
        key_value = _find_by_label(keys, target)
        if not key_value:
            sys.exit(f"error: no key with label or value {target!r}")
    label = (keys[key_value] or {}).get("label", "?")
    del keys[key_value]
    _save(args.file, keys)
    print(f"Revoked key for {label!r} ({key_value[:12]}…)")
    return 0


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="manage_keys",
        description="Mint, list, and revoke ARSE API keys.",
    )
    p.add_argument("--file", type=Path, default=DEFAULT_PATH,
                   help=f"Path to the keys JSON (default: {DEFAULT_PATH}).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_add = sub.add_parser("add", help="Mint a new key for a labelled user.")
    p_add.add_argument("label",
                       help="Human-readable identifier (a name, "
                            "service handle, or anything searchable).")
    p_add.add_argument("--expires",
                       help="Optional expiry as ISO-8601 (YYYY-MM-DD "
                            "or YYYY-MM-DDTHH:MM:SS, UTC unless an offset "
                            "is given).  No expiry by default.")
    p_add.set_defaults(func=cmd_add)

    p_list = sub.add_parser("list", help="List all keys with their metadata.")
    p_list.set_defaults(func=cmd_list)

    p_revoke = sub.add_parser("revoke",
                              help="Revoke a key by label or by value.")
    p_revoke.add_argument("target",
                          help="Label (e.g. 'alice') or full key value.")
    p_revoke.set_defaults(func=cmd_revoke)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
