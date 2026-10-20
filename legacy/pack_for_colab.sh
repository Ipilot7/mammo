#!/bin/sh
# Pack project code (without data/venv/weights) for upload to Google Colab.
cd "$(dirname "$0")/.." && rm -f mammo_code.zip && \
  zip -qr mammo_code.zip mammo train scripts app requirements.txt requirements-serve.txt -x '*/__pycache__/*' && \
  echo "created mammo_code.zip"
