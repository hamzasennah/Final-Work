#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-rpi.txt
sudo cp agroshield-rpi.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable agroshield-rpi
sudo systemctl restart agroshield-rpi
