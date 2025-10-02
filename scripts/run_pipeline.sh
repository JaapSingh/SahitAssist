#!/usr/bin/env bash
set -euo pipefail

BOOK_ID=${1:-book1}
RAW_TXT="input/${BOOK_ID}_raw.txt"
OUT_DIR="outputs/${BOOK_ID}"

echo "Running SahitAssist pipeline for ${BOOK_ID}"
sahitassist pipeline --anmol-text "${RAW_TXT}" --book-id "${BOOK_ID}" --output "${OUT_DIR}" --use-context