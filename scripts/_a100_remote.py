"""Helper for SSH operations on A100 instances."""
from __future__ import annotations

import sys

import paramiko

HOST = "fj02-ssh.gpuhome.cc"
PORT = 30182
USER = "root"
PASSWORD = "b2z8k8xg"


def connect() -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30, banner_timeout=30)
    return ssh


def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 30) -> tuple[str, str]:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return out, err


def check_dolly(ssh: paramiko.SSHClient) -> str:
    out, err = run(ssh, "python3 /root/check_dolly.py 2>&1 || echo NO_DOLLY")
    return out


if __name__ == "__main__":
    ssh = connect()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "echo ok"
    out, err = run(ssh, cmd)
    if out:
        print(out)
    if err:
        print("STDERR:", err[:500])
    ssh.close()
