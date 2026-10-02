#!/bin/sh
# Starts a release: sets the version everywhere, commits, tags and pushes.
# The Release workflow then builds and publishes everything (crates.io, PyPI,
# npm, container image, binaries and the GitHub release).
#
#   scripts/release.sh 0.2.0
set -eu
version="${1:?usage: scripts/release.sh X.Y.Z}"
case "$version" in
  [0-9]*.[0-9]*.[0-9]*) ;;
  *) echo "version must look like X.Y.Z" >&2; exit 1 ;;
esac
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain)" ]; then
  echo "commit or stash your changes first" >&2
  exit 1
fi
old=$(sed -n '/^\[workspace.package\]/,/^\[/s/^version = "\(.*\)"/\1/p' Cargo.toml)
# Workspace version and the versions of the internal dependencies.
sed -i.bak -e "/^\[workspace.package\]/,/^\[/s/^version = \"$old\"/version = \"$version\"/" \
  -e "s/\(^tomoz-[a-z]* = { path = \"crates\/tomoz-[a-z]*\", version = \"\)$old\"/\1$version\"/" Cargo.toml
rm Cargo.toml.bak
(cd crates/tomoz-wasm/js && npm version "$version" --no-git-tag-version > /dev/null)
cargo update --workspace --offline > /dev/null 2>&1 || cargo update --workspace > /dev/null
git add Cargo.toml Cargo.lock crates/tomoz-wasm/js/package.json
git commit -m "chore(release): $version"
git tag -a "v$version" -m "Tomoz $version"
git push origin HEAD "v$version"
echo "v$version pushed; follow the Release workflow on GitHub"
