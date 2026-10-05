import base64
import ipaddress
import re
import stat

from cryptography import x509

from reachy_muse.companion_server.pair import pair


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_pair_writes_private_state_and_a_pinnable_line(tmp_path):
    state = tmp_path / "state"
    paired = pair(state, hosts=["studio.local"])
    assert mode(state) == 0o700
    assert {p.name: mode(p) for p in state.iterdir()} == {"cert.pem": 0o600, "key.pem": 0o600, "token": 0o600}
    assert re.fullmatch(r"wss://studio\.local:8765  pin=sha256:[0-9a-f]{64}  token=rc1_[A-Za-z0-9_-]{43}", paired.line("studio.local"))
    assert len(base64.urlsafe_b64decode(paired.token[4:] + "=")) == 32
    names = x509.load_pem_x509_certificate(paired.cert_pem).extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert (names.get_values_for_type(x509.DNSName), names.get_values_for_type(x509.IPAddress)) == (
        ["localhost", "studio.local"],
        [ipaddress.ip_address("127.0.0.1")],
    )


def test_pair_again_keeps_the_pairing_and_force_replaces_it(tmp_path):
    first = pair(tmp_path, hosts=["studio.local"])
    again = pair(tmp_path, hosts=["other.local"])
    forced = pair(tmp_path, hosts=["studio.local"], force=True)
    assert (again.pin, again.token) == (first.pin, first.token)
    assert forced.pin != first.pin and forced.token != first.token


def test_pair_fills_in_a_missing_token_without_touching_the_certificate(tmp_path):
    first = pair(tmp_path, hosts=["studio.local"])
    (tmp_path / "token").unlink()
    healed = pair(tmp_path, hosts=["studio.local"])
    assert healed.pin == first.pin and healed.token != first.token


def test_pair_replaces_a_certificate_that_does_not_match_its_key(tmp_path):
    first = pair(tmp_path, hosts=["studio.local"])
    (tmp_path / "key.pem").write_bytes(pair(tmp_path / "other", hosts=["x"]).key_path.read_bytes())
    (tmp_path / "token").chmod(0o644)
    healed = pair(tmp_path, hosts=["studio.local"])
    assert (healed.pin != first.pin, healed.token == first.token, mode(tmp_path / "token")) == (True, True, 0o600)
    healed.tls_context()
