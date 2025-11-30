#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

# Parse command line arguments
SKIP_TESTS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-tests    Skip all test checks"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done


# Print functions
stdmsg() {
  local IFS=' '
  printf '%s\n' "$*"
}

errmsg() {
  stdmsg "$*" 1>&2
}

# Trap exit handler
trap_exit() {
  # It is critical that the first line capture the exit code. Nothing
  # else can come before this. The exit code recorded here comes from
  # the command that caused the script to exit.
  local exit_status="$?"

  if [[ ${exit_status} -ne 0 ]]; then
    errmsg 'The script did not complete successfully.'
    errmsg 'The exit code was '"${exit_status}"
  fi
}
trap trap_exit EXIT

base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd -P)"
project_dir="$(cd "${base_dir}/.." >/dev/null && pwd -P)"

stdmsg $project_dir
stdmsg $base_dir

# cd to the directory before running rye
cd "${project_dir}"

stdmsg "Running uv sync with dev dependencies..."
uv sync --dev

stdmsg "Checking uv.lock"
uv lock --check

stdmsg "Checking Lint and Format"
uv run ruff check .
uv run ruff format --check .

stdmsg "Type checking with pyright"
uv run pyright src/inspeqtor/.

if [[ $SKIP_TESTS == false ]]; then
stdmsg "Running test with python 3.13 3.12 and 3.11"
uv run --python 3.13 --isolated --with-editable '.[test]' pytest tests/.
uv run --python 3.12 --isolated --with-editable '.[test]' pytest tests/.
uv run --python 3.11 --isolated --with-editable '.[test]' pytest tests/.
else
stdmsg "Skip the test"
fi

stdmsg "Checking build"
uv build

