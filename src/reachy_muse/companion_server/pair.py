"""Pairing state: a self-signed certificate the robot pins, and the token it sends in hello."""

from __future__ import annotations

import base64
import datetime
import hashlib
import ipaddress
import os
import secrets
import socket
import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

DEFAULT_PORT = 8765
TOKEN_BYTES = 32
CERT_DAYS = 3650


def default_state_dir() -> Path:
    return Path(os.environ.get("REACHY_COMPANION_HOME", "~/.reachy-companion")).expanduser()


@dataclass(frozen=True)
class Pairing:
    state_dir: Path
    cert_pem: bytes
    token: str

    @property
    def cert_path(self) -> Path:
        return self.state_dir / "cert.pem"

    @property
    def key_path(self) -> Path:
        return self.state_dir / "key.pem"

    @property
    def pin(self) -> str:
        """SHA-256 of the DER certificate, which is what the robot compares after the TLS handshake."""
        return "sha256:" + hashlib.sha256(ssl.PEM_cert_to_DER_cert(self.cert_pem.decode())).hexdigest()

    def line(self, host: str, port: int = DEFAULT_PORT) -> str:
        return f"wss://{host}:{port}  pin={self.pin}  token={self.token}"

    def tls_context(self) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(self.cert_path, self.key_path)
        return context


def pair(state_dir: Path, *, hosts: Iterable[str], force: bool = False) -> Pairing:
    """Create whatever is missing, or everything with force. Re-running keeps an existing pairing."""
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    state_dir.chmod(0o700)
    cert, key, token = state_dir / "cert.pem", state_dir / "key.pem", state_dir / "token"
    if force or not _usable(cert, key):
        cert_pem, key_pem = _self_signed(list(hosts))
        _write(key, key_pem)
        _write(cert, cert_pem)
    if force or not token.exists():
        _write(token, ("rc1_" + base64.urlsafe_b64encode(secrets.token_bytes(TOKEN_BYTES)).rstrip(b"=").decode()).encode())
    for path in (cert, key, token):
        path.chmod(0o600)
    return load(state_dir)


def _usable(cert: Path, key: Path) -> bool:
    """False for a missing file or a key from a run that crashed before writing its certificate."""
    try:
        ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(cert, key)
    except (OSError, ssl.SSLError):
        return False
    return True


def load(state_dir: Path) -> Pairing:
    try:
        return Pairing(state_dir, (state_dir / "cert.pem").read_bytes(), (state_dir / "token").read_text().strip())
    except FileNotFoundError:
        raise SystemExit(f"{state_dir} has no pairing. Run: reachy-companion pair") from None


def advertised_host() -> str:
    """The Mac's Bonjour name, which survives DHCP changes better than an IP."""
    name = socket.gethostname()
    return name if "." in name else f"{name}.local"


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as out:
        out.write(data)
    os.replace(tmp, path)


def _self_signed(hosts: list) -> tuple:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "reachy-companion")])
    now = datetime.datetime.now(datetime.timezone.utc)
    alt_names = []
    for host in dict.fromkeys(["localhost", "127.0.0.1", *hosts]):
        try:
            alt_names.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            alt_names.append(x509.DNSName(host))
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=CERT_DAYS))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .sign(key, hashes.SHA256())
    )
    key_pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return cert.public_bytes(serialization.Encoding.PEM), key_pem
