#!/usr/bin/env bash
# Install the inference binary and systemd unit. Run on the Raspberry Pi.
# Usage: sudo ./scripts/install.sh /path/to/model.bin
set -euo pipefail

MODEL=${1:?usage: install.sh MODEL.bin}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PREFIX=/opt/potok-ai

if [[ ! -f "$MODEL" ]]; then
    echo "model file not found: $MODEL" >&2
    exit 1
fi
if [[ ! -x "$ROOT/target/release/infer" ]]; then
    echo "missing $ROOT/target/release/infer — run 'cargo build --release' first" >&2
    exit 1
fi

install -d -m 0755 "$PREFIX"
install -m 0755 "$ROOT/target/release/infer" "$PREFIX/infer"
install -m 0644 "$MODEL" "$PREFIX/model.bin"
install -m 0644 "$ROOT/README.md" "$PREFIX/README.md"
install -m 0644 "$ROOT/systemd/potok-infer.service" /etc/systemd/system/potok-infer.service

if ! id potok >/dev/null 2>&1; then
    useradd --system --shell /usr/sbin/nologin --home-dir /nonexistent potok
fi
if getent group dialout >/dev/null 2>&1; then
    usermod -aG dialout potok || true
fi

systemctl daemon-reload
systemctl enable potok-infer.service
echo "Installed. Start with: systemctl start potok-infer.service"
echo "Check with: journalctl -u potok-infer.service -f"
