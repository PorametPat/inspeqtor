#!/usr/bin/env bash

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

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

# Colors for output
readonly RED=$'\e[38;5;196m'
readonly GREEN=$'\e[38;5;76m'
readonly YELLOW=$'\e[38;5;226m'
readonly BLUE=$'\e[38;5;33m'
readonly NC=$'\e[0m'

# Counters
declare -i total_checks=0
declare -i passed_checks=0
declare -i failed_checks=0

# Function to print section headers
print_header() {
    echo ""
    echo "${BLUE}$1${NC}"
}

# Function to print success
print_success() {
    ((passed_checks++))
    echo "${GREEN}✓${NC} $1"
}

# Function to print error
print_error() {
    ((failed_checks++))
    echo "${RED}✗${NC} $1"
}

# Function to print info
print_info() {
    echo "${BLUE}·${NC} $1"
}

# Function to print skipped
print_skipped() {
    echo "${YELLOW}⊘${NC} $1"
}

# Function to run a check with timing
run_check() {
    local check_name="$1"
    shift
    local start_time
    start_time=$(date +%s%N)
    
    ((total_checks++))
    
    if "$@" &> /dev/null; then
        local end_time
        end_time=$(date +%s%N)
        local duration=$(( (end_time - start_time) / 1000000 ))
        print_success "$check_name (${duration}ms)"
    else
        print_error "$check_name"
        return 1
    fi
}

# Cleanup on exit
cleanup() {
    echo ""
    if [[ $failed_checks -eq 0 ]]; then
        echo "${GREEN}✓ All checks passed${NC} ($passed_checks/$total_checks)"
    else
        echo "${RED}✗ Some checks failed${NC} (${GREEN}$passed_checks${NC}/${RED}$failed_checks${NC}/$total_checks)"
    fi
}

trap cleanup EXIT
trap 'print_error "Check failed!"; exit 1' ERR

print_header "Code Quality Checks"

# 1. Lock file check
run_check "Lock file" uv lock --locked

# 2. Linting
run_check "Linting" uvx ruff check .
run_check "Formatting" uvx ruff format --check .
run_check "Type checking" uv run pyright src/inspeqtor/.

# 3. Testing
if [[ $SKIP_TESTS == false ]]; then
    run_check "Tests (Python 3.13)" uv run --python 3.13 --isolated --with-editable '.[test]' pytest tests/.
    run_check "Tests (Python 3.12)" uv run --python 3.12 --isolated --with-editable '.[test]' pytest tests/.
    run_check "Tests (Python 3.11)" uv run --python 3.11 --isolated --with-editable '.[test]' pytest tests/.
else
    print_skipped "Tests (Python 3.13)"
    print_skipped "Tests (Python 3.12)"
    print_skipped "Tests (Python 3.11)"
fi

# 4. Build
run_check "Build" uv build

