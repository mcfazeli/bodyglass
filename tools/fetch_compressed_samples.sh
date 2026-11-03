#!/usr/bin/env bash
# Download the public compressed multi-frame DICOM samples used by tests/test_compressed_multiframe.py.
# Every file is pinned to a commit and checked by SHA-256; they are only ever read as DICOM data. Sources and licences:
#   pydicom-data  https://github.com/pydicom/pydicom-data (MIT)
#   pydicom       https://github.com/pydicom/pydicom test files (MIT)
#   GDCM data     https://github.com/malaterre/gdcmdata (GDCM BSD-3-Clause licence, as quoted in pydicom's test-file README)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p sample_data/compressed_multiframe && cd sample_data/compressed_multiframe
PYD=https://raw.githubusercontent.com/pydicom/pydicom-data/abc42b90985fb6cf385aa4af766d2c9c94a257a4/data_store/data
GDCM=https://raw.githubusercontent.com/malaterre/gdcmdata/9b38ac7a8baee4e943505ab3c56fb7d19d54754f
PYT=https://raw.githubusercontent.com/pydicom/pydicom/413945fee751a4260c567e1bf8377adfc37ba6ba/src/pydicom/data/test_files
get() { [ -f "$1" ] || curl -L --fail --retry 3 -sS -o "$1" "$2"; }
get color3d_jpeg_baseline.dcm            "$PYD/color3d_jpeg_baseline.dcm"
get emri_small.dcm                       "$PYD/emri_small.dcm"
get emri_small_jpeg_2k_lossless.dcm      "$PYD/emri_small_jpeg_2k_lossless.dcm"
get EnhancedWithIPPPerFrameIOPShared.dcm "$GDCM/EnhancedWithIPPPerFrameIOPShared.dcm"
get D_CLUNIE_CT1_J2KI.dcm                "$GDCM/D_CLUNIE_CT1_J2KI.dcm"
get 693_J2KI.dcm                         "$PYT/693_J2KI.dcm"
sha256sum -c --quiet <<'SUMS'
c8798b8abf8ae0a18e8c9952e7c7f75f3cc8465234b1b63f9e3ba3bebb9d5625  color3d_jpeg_baseline.dcm
151233ec63f64ebb63b979df51aa827cd612a53422c073f6ef341770c7bc9a56  emri_small.dcm
b2b4063359a08ed3b0afa9f4e4f72f84af79e5116515b446d9a30da9dc7f1888  emri_small_jpeg_2k_lossless.dcm
535679f02f157f78072ad8cb5eec20705db616924b6634bca67c28e419789cc8  EnhancedWithIPPPerFrameIOPShared.dcm
ee5cb9fa931ac6d31a67cd8b58d05d7e6d5212f4666ba41c2b3ece9a069c2738  D_CLUNIE_CT1_J2KI.dcm
8d5d503fd46b9a59c628762d71d7391ea1a2a5fd8d339ac82ef9e281a15ef65f  693_J2KI.dcm
SUMS
