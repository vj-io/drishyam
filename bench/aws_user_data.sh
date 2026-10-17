#!/bin/bash
# EC2 user-data for one benchmark arm (bench/aws_runbook.md); runs once, as root, at first boot.
# Installs git, the AWS CLI and uv, clones the public repo and arms a hard stop: the instance is launched with
# shutdown behaviour "terminate", so the `shutdown` below ends it, root volume included, after 3 hours.
set -euxo pipefail
exec >>/var/log/drishyam-setup.log 2>&1
export HOME=/root  # cloud-init does not always set it, and the uv installer needs it
shutdown -h +180
if ! command -v git >/dev/null; then
  if command -v apt-get >/dev/null; then apt-get update -q && apt-get install -yq git; else dnf install -yq git; fi
fi
if ! command -v aws >/dev/null; then  # Ubuntu AMIs ship snapd but no AWS CLI; Amazon Linux ships the CLI
  snap wait system seed.loaded && snap install aws-cli --classic
fi
curl -LsSf https://astral.sh/uv/0.13.1/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
git clone -q https://github.com/vj-io/drishyam /opt/drishyam
touch /var/lib/drishyam-ready
