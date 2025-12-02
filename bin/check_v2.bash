#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

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

# Parse command line arguments
INSPEQTOR_SKIP_TESTS=${INSPEQTOR_SKIP_TESTS:-false}
if [[ ${INSPEQTOR_SKIP_TESTS} != 'true' ]] && [[ ${INSPEQTOR_SKIP_TESTS} != 'false' ]]; then
  errmsg "The environment variable INSPEQTOR_SKIP_TESTS was set to the invalid value '${INSPEQTOR_SKIP_TESTS}'. Valid values are 'true' and 'false'."
  exit 64
fi
while [[ $# -gt 0 ]]; do
  case $1 in
  --skip-tests)
    INSPEQTOR_SKIP_TESTS=true
    shift
    ;;
  --help | -h)
    stdmsg "Usage: $0 [OPTIONS]"
    stdmsg ""
    stdmsg "Options:"
    stdmsg "  --skip-tests    Skip all test checks. Tests can also be skipped by setting the INSPEQTOR_SKIP_TESTS environment variable to 'true'. The command-line option overrides the environment variable."
    stdmsg "  --help          Show this help message and exit."
    exit 0
    ;;
  *)
    errmsg "Unknown option: $1"
    errmsg "Run with --help for usage information"
    exit 64
    ;;
  esac
done

stdmsg "${project_dir}"
stdmsg "${base_dir}"

# cd to the project directory before running any other operations
cd "${project_dir}"

stdmsg "Running uv sync with dev dependencies..."
uv sync --dev

stdmsg "Checking uv.lock"
uv lock --check

stdmsg "Checking Lint and Format"
uv run ruff check .
uv run ruff format --check .

stdmsg "Type checking with pyright"
uv run pyright src/inspeqtor

if [[ ${INSPEQTOR_SKIP_TESTS} == false ]]; then
  stdmsg "Running test suite for all supported Python versions"
  while IFS='' read -r version; do
    uv run --python "${version}" --isolated --with-editable '.[test]' pytest tests/.
  done <<EOF
3.13
3.12
3.11
EOF
else
  stdmsg "Skipping tests"
fi

stdmsg "Building distribution package"
uv build
