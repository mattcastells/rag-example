#!/usr/bin/env bash
set -euo pipefail

if [ ! -f ../.env ]; then
  cp ../.env.example ../.env
  echo ".env creado. Actualiza ANTHROPIC_API_KEY antes de continuar."
fi

poetry install
