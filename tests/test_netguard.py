import socket

import pytest

import netguard


def _resolve_to(monkeypatch, ip):
    monkeypatch.setattr(socket, "getaddrinfo",
                        lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))])


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.5", "172.18.0.3", "192.168.1.1",
                                "169.254.169.254", "100.64.0.1", "0.0.0.0", "::1", "::ffff:127.0.0.1"])
def test_private_addresses_blocked(monkeypatch, ip):
    _resolve_to(monkeypatch, ip)
    with pytest.raises(netguard.FetchError) as e:
        netguard.check_url("http://example.org/")
    assert e.value.kind == "blocked"


@pytest.mark.parametrize("url", ["ftp://example.org/", "file:///etc/passwd",
                                 "http://example.org:8070/", "http://user:pw@example.org/",
                                 "gopher://example.org/"])
def test_bad_urls_blocked(url):
    with pytest.raises(netguard.FetchError) as e:
        netguard.check_url(url)
    assert e.value.kind == "blocked"


def test_public_address_allowed(monkeypatch):
    _resolve_to(monkeypatch, "93.184.216.34")
    netguard.check_url("https://example.org/page")


def test_unresolvable_is_dns_error(monkeypatch):
    def fail(*a, **k):
        raise socket.gaierror("nope")
    monkeypatch.setattr(socket, "getaddrinfo", fail)
    with pytest.raises(netguard.FetchError) as e:
        netguard.check_url("https://nope.invalid/")
    assert e.value.kind == "dns"


class _Resp:
    def __init__(self, status, headers=None, body=b""):
        self.status_code, self.headers, self._body = status, headers or {}, body
    def iter_content(self, chunk_size):
        yield self._body
    def close(self):
        pass


def test_redirect_into_private_network_blocked(monkeypatch):
    def resolve(host, *a, **k):
        ip = "10.0.0.1" if host == "internal.example" else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))]
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(netguard._SESSION, "get",
                        lambda url, **k: _Resp(302, {"location": "http://internal.example/admin"}))
    with pytest.raises(netguard.FetchError) as e:
        netguard.safe_get("https://public.example/")
    assert e.value.kind == "blocked"


def test_body_is_capped(monkeypatch):
    _resolve_to(monkeypatch, "93.184.216.34")
    monkeypatch.setattr(netguard._SESSION, "get",
                        lambda url, **k: _Resp(200, {"Content-Type": "text/html"}, b"x" * 5000))
    r = netguard.safe_get("https://example.org/", max_bytes=1000)
    assert r.truncated and len(r.content) == 1000 and r.ok
