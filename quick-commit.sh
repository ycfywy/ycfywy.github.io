#!/bin/bash
set -e

MSG="$(date '+%Y-%m-%d %H:%M')"

git add .
git commit -m "$MSG"
git push
