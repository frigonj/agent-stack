#!/bin/bash
# Description: Search all source files in /workspace/src and /workspace/repos for a keyword or pattern
# Usage: ./search-codebase.sh <pattern> [directory]
# Tags: grep, search, code, codebase, pattern
set -euo pipefail

PATTERN="${1:?Usage: search-codebase.sh <pattern> [directory]}"
DIR="${2:-/workspace/src}"

echo "=== Searching '$PATTERN' in $DIR ==="
grep -rn --include="*.py" --include="*.yml" --include="*.yaml" --include="*.sh" \
     --color=never \
     "$PATTERN" "$DIR" 2>/dev/null | head -100

echo ""
echo "(showing up to 100 matches)"
