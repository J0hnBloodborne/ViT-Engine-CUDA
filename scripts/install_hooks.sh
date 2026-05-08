#!/usr/bin/env bash
set -e
REPO_ROOT=$(git rev-parse --show-toplevel)
git config core.hooksPath .githooks
chmod +x .githooks/pre-push || true
echo "Installed hooks: git core.hooksPath set to .githooks"
