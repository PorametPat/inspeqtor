#!/usr/bin/env bash

# Usage: release.bash [next-version]
# If a version number is not provided, the next version will be a patch version increment.

set -euo pipefail

# Colors for output
readonly RED=$'\e[38;5;196m'
readonly GREEN=$'\e[38;5;76m'
readonly YELLOW=$'\e[38;5;226m'
readonly BLUE=$'\e[38;5;33m'
readonly NC=$'\e[0m'

# Print functions
stdmsg() {
    echo "$*"
}

errmsg() {
    echo -e "${RED}Error:${NC} $*" >&2
}

info() {
    echo -e "${BLUE}·${NC} $*"
}

success() {
    echo -e "${GREEN}✓${NC} $*"
}

# Get the base and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Trap exit handler
trap_exit() {
    local exit_status=$?
    if [[ ${exit_status} -ne 0 ]]; then
        errmsg "The script did not complete successfully (exit code: ${exit_status})"
    fi
    exit ${exit_status}
}
trap trap_exit EXIT

# Ensure that the script is run from the main branch
info "Checking git state..."
current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ ${current_branch} != "main" ]]; then
    errmsg "This script must be run on the main branch. Current branch: ${current_branch}"
    exit 1
fi
success "On main branch"

# Ensure that there are no uncommitted changes
if ! git diff-index --quiet HEAD --; then
    errmsg "There are uncommitted changes. Please commit or stash them before running this script."
    exit 1
fi
success "No uncommitted changes"

# Get current version from pyproject.toml
info "Reading current version from pyproject.toml..."
current_version=$(grep '^version = ' pyproject.toml | grep -oP '(?<=")[^"]*(?=")')

# Remove `.dev0` from the version if present
updated_version=${current_version%.dev0}
updated_version=${updated_version%.dev}

# Check format of updated_version
if [[ ! ${updated_version} =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    errmsg "Current version '${updated_version}' is not in the correct format (A.B.C)"
    exit 1
fi
success "Current version: ${updated_version}"

# Parse optional next version argument
new_version=""

if [[ -n ${1-} ]]; then
    if [[ $1 =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        new_version="$1"
    else
        errmsg "Invalid next version format. Use A.B.C (e.g., 1.2.3)"
        exit 1
    fi
else
    # No argument provided, increment patch version
    IFS='.' read -r major minor patch <<<"${updated_version}"
    new_version="${major}.${minor}.$((patch + 1))"
fi
info "Next version will be: ${new_version}"

# Run the checks script
info "Running checks before starting release process..."
"${SCRIPT_DIR}/check.bash" --skip-tests || exit 1

# Update version in pyproject.toml
info "Updating version in pyproject.toml to ${updated_version}..."
sed -i.bak "s/^version = .*/version = \"${updated_version}\"/" pyproject.toml
rm -f pyproject.toml.bak
success "Version updated"

# Create release branch
branch_name="release-${updated_version}"
info "Creating release branch '${branch_name}'..."
git checkout -b "${branch_name}" || exit 1
success "Branch created"

# Commit the version change
info "Committing version change..."
git add pyproject.toml
git commit -m "Release version: ${updated_version}" || exit 1
git push origin "${branch_name}" || exit 1
success "Changes pushed"

# Create a tag
info "Creating tag 'v${updated_version}'..."
git tag -am "Release version: ${updated_version}" "v${updated_version}" || exit 1
git push origin "v${updated_version}" || exit 1
success "Tag created and pushed"

# Update documentation with mike
info "Updating documentation with mike..."
uv run mike deploy --push "${updated_version}" --update-aliases latest || exit 1
success "Documentation updated"

# Set latest as the default version
info "Setting 'latest' as the default documentation version..."
uv run mike set-default --push latest || exit 1
success "Default documentation version set"

# Update the version to the next dev version
info "Starting next version: ${new_version}.dev0..."
sed -i.bak "s/^version = .*/version = \"${new_version}.dev0\"/" pyproject.toml
rm -f pyproject.toml.bak
success "Version updated to dev"

# Commit the next version change
info "Committing next version change..."
git add pyproject.toml
git commit -m "Start next version: ${new_version}.dev0" || exit 1
git push origin "${branch_name}" || exit 1
success "Next version committed"

# Create a pull request
pull_request_url="https://github.com/PorametPat/inspeqtor/pull/new/${branch_name}"
stdmsg ""
success "Release completed successfully!"
info "Please review and merge the pull request:"
info "${pull_request_url}"

if command -v xdg-open &>/dev/null; then
    xdg-open "${pull_request_url}" 2>/dev/null || true
elif command -v open &>/dev/null; then
    open "${pull_request_url}" 2>/dev/null || true
fi
