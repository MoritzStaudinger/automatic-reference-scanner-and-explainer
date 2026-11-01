"""
SSRF-safe HTTP fetching for URLs that come from user-uploaded PDFs.

Citations carry arbitrary URLs, and the deployed instance is public, so a
crafted PDF could otherwise make the server fetch `http://grobid:8070/…`,
`http://arse-api:8500/…`, cloud metadata endpoints (169.254.169.254) or
anything else on the internal network — and echo the page title back in
the UI.  ``safe_get`` refuses that:

  * only http / https, only ports 80 / 443
  * every hostname is resolved and *all* its addresses must be globally
    routable (no private, loopback, link-local, multicast, reserved, or
    shared-address-space IPs; no IPv4-mapped tricks)
  * redirects are followed manually (≤ 5 hops) and every hop is re-checked
  * the body is streamed and truncated at ``max_bytes``

Known limitation: the name is resolved once for the check and again by
`requests` for the connection, so a DNS-rebinding attacker with a very
short TTL could still race it.  Internal Docker service names never
resolve publicly, which is the realistic threat here.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit

import requests

MAX_REDIRECTS = 5
_ALLOWED_PORTS = {None, 80, 443}

# Separate session: no shared connection pool with the academic-API
# session, and never any proxy/auth configuration leaking into it.
_SESSION = requests.Session()
_SESSION.trust_env = False


class FetchError(Exception):
    """``kind`` is one of: blocked, dns, connect, timeout, too_many_redirects."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


@dataclass
class FetchResult:
    url: str                       # final URL after redirects
    status_code: int
    headers: dict[str, str]
    content: bytes
    truncated: bool = False
    history: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def content_type(self) -> str:
        return (self.headers.get("content-type") or "").lower()

    @property
    def text(self) -> str:
        enc = "utf-8"
        ct = self.content_type
        if "charset=" in ct:
            enc = ct.split("charset=", 1)[1].split(";")[0].strip() or enc
        try:
            return self.content.decode(enc, errors="replace")
        except LookupError:
            return self.content.decode("utf-8", errors="replace")


def _is_public_ip(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast


def check_url(url: str) -> None:
    """Raise FetchError('blocked' | 'dns') unless ``url`` is safe to fetch."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise FetchError("blocked", f"scheme {parts.scheme!r} not allowed")
    host = parts.hostname
    if not host:
        raise FetchError("blocked", "URL has no host")
    try:
        port = parts.port
    except ValueError:
        raise FetchError("blocked", "invalid port") from None
    if port not in _ALLOWED_PORTS:
        raise FetchError("blocked", f"port {port} not allowed")
    if parts.username or parts.password:
        raise FetchError("blocked", "credentials in URL not allowed")
    try:
        infos = socket.getaddrinfo(host, port or (443 if parts.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError) as e:
        raise FetchError("dns", f"cannot resolve {host}: {e}") from None
    addrs = {info[4][0] for info in infos}
    if not addrs or not all(_is_public_ip(a) for a in addrs):
        raise FetchError("blocked", f"{host} resolves to a non-public address")


def safe_get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 10,
    max_bytes: int = 512 * 1024,
) -> FetchResult:
    """GET ``url`` with SSRF protection.  Raises FetchError on refusal or
    transport failure; HTTP error statuses are returned, not raised."""
    history: list[str] = []
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        check_url(current)
        try:
            r = _SESSION.get(current, headers=headers, timeout=timeout,
                             allow_redirects=False, stream=True)
        except requests.Timeout as e:
            raise FetchError("timeout", str(e)[:200]) from None
        except requests.RequestException as e:
            raise FetchError("connect", str(e)[:200]) from None
        try:
            loc = r.headers.get("location")
            if r.status_code in (301, 302, 303, 307, 308) and loc:
                history.append(current)
                current = urljoin(current, loc)
                continue
            chunks: list[bytes] = []
            size = 0
            truncated = False
            for chunk in r.iter_content(chunk_size=16384):
                chunks.append(chunk)
                size += len(chunk)
                if size >= max_bytes:
                    truncated = True
                    break
            body = b"".join(chunks)[:max_bytes]
            return FetchResult(
                url=current,
                status_code=r.status_code,
                headers={k.lower(): v for k, v in r.headers.items()},
                content=body,
                truncated=truncated,
                history=history,
            )
        finally:
            r.close()
    raise FetchError("too_many_redirects", f"more than {MAX_REDIRECTS} redirects")
