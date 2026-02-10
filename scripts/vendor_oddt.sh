#!/usr/bin/env bash

set -euo pipefail

REPO_URL="${ODDT_VENDOR_REPO:-https://github.com/Arturossi/oddt.git}"
REPO_REF="${ODDT_VENDOR_REF:-6e5629ed05aa4931421b14cace8f2fea5bff5bb6}"
DEST_DIR="${ODDT_VENDOR_DEST:-oddt}"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

echo "Vendoring ODDT from ${REPO_URL}@${REPO_REF} into ${DEST_DIR}"

git -C "${tmp_dir}" init -q
git -C "${tmp_dir}" remote add origin "${REPO_URL}"
git -C "${tmp_dir}" fetch --depth 1 origin "${REPO_REF}"
git -C "${tmp_dir}" checkout -q FETCH_HEAD

if [[ ! -d "${tmp_dir}/oddt" ]]; then
  echo "ERROR: Expected '${tmp_dir}/oddt' package directory was not found." >&2
  exit 1
fi

rm -rf "${DEST_DIR}"
cp -a "${tmp_dir}/oddt" "${DEST_DIR}"

if [[ -f "${tmp_dir}/LICENSE" ]]; then
  cp -f "${tmp_dir}/LICENSE" "${DEST_DIR}/LICENSE.vendored"
fi

if [[ -f "${tmp_dir}/README.md" ]]; then
  cp -f "${tmp_dir}/README.md" "${DEST_DIR}/README.vendored.md"
fi

echo "${REPO_URL}@${REPO_REF}" > "${DEST_DIR}/.vendored-ref"
echo "Vendored ODDT successfully."
