#!/bin/bash
# Authenticode-sign one or more Windows PE files (.exe/.dll) with osslsigncode,
# mirroring the LAMMPS-GUI packaging/sign.sh. Signing is a graceful no-op unless
# a code-signing certificate is actually provided, so unsigned CI builds (forks,
# pull requests, contributors without the certificate) keep working.
#
# Usage: sign.sh FILE [FILE ...]
#
# Environment:
#   SIGN_DISABLE   set to 1 to skip signing entirely
#   SIGN_PFX       path to the PKCS#12 (.pfx) code-signing certificate
#                  (default: $HOME/.codesign/sparta-gui.pfx). If the file does
#                  not exist, signing is skipped (files are left unsigned).
#   SIGN_PASSWORD  password for the .pfx (required once a certificate is present)
#   SIGN_NAME      description embedded in the signature (default: SPARTA-GUI)
#   SIGN_URL       URL embedded in the signature
#                  (default: https://sparta.github.io/sparta-gui)
#   SIGN_TSA       RFC3161/Authenticode timestamp server
#                  (default: http://timestamp.digicert.com)
#
# A cloud HSM / token can be used instead of a .pfx by replacing the
# "-pkcs12 ... -pass ..." arguments below with osslsigncode's PKCS#11 engine
# options (-pkcs11engine/-pkcs11module/-certs/-key) for the provider.
set -e

SIGN_PFX="${SIGN_PFX:-$HOME/.codesign/sparta-gui.pfx}"
SIGN_NAME="${SIGN_NAME:-SPARTA-GUI}"
SIGN_URL="${SIGN_URL:-https://sparta.github.io/sparta-gui}"
SIGN_TSA="${SIGN_TSA:-http://timestamp.digicert.com}"

if [[ "${SIGN_DISABLE:-0}" == "1" ]]; then
  echo "sign.sh: signing disabled (SIGN_DISABLE=1); leaving files unsigned"
  exit 0
fi
if [[ ! -f "$SIGN_PFX" ]]; then
  echo "sign.sh: no code-signing certificate at '$SIGN_PFX'; leaving files unsigned"
  exit 0
fi
if [[ -z "${SIGN_PASSWORD:-}" ]]; then
  echo "sign.sh: SIGN_PASSWORD is not set but a certificate is present; cannot sign" >&2
  exit 1
fi
if ! command -v osslsigncode >/dev/null 2>&1; then
  echo "sign.sh: osslsigncode not found in PATH" >&2
  exit 1
fi

for f in "$@"; do
  if [[ ! -f "$f" ]]; then
    echo "sign.sh: skipping missing file '$f'"
    continue
  fi
  tmp="$(mktemp)"
  echo "sign.sh: signing $f"
  osslsigncode sign \
      -pkcs12 "$SIGN_PFX" -pass "$SIGN_PASSWORD" \
      -n "$SIGN_NAME" -i "$SIGN_URL" \
      -h sha256 -ts "$SIGN_TSA" \
      -in "$f" -out "$tmp"
  mv -f "$tmp" "$f"
  osslsigncode verify -in "$f" >/dev/null && echo "sign.sh: verified $f"
done
