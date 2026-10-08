#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Keep graphics fallback local to this process; do not change the desktop setup.
if [[ "${POINT_CLOUD_STUDIO_RENDER:-auto}" == "software" ]] || \
   { [[ "${POINT_CLOUD_STUDIO_RENDER:-auto}" == "auto" ]] && command -v glxinfo >/dev/null && \
     ! timeout 5 glxinfo -B >/dev/null 2>&1; }; then
  export __GLX_VENDOR_LIBRARY_NAME=mesa
  export LIBGL_ALWAYS_SOFTWARE=1
  unset LD_LIBRARY_PATH
fi
exec /usr/bin/python3 viewer.py "$@"
