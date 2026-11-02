#!/usr/bin/env bash
# Download the public compressed multi-frame DICOM samples used by tests/test_compressed_multiframe.py.
# Every file is pinned to a commit and checked by SHA-256. Sources and licences:
#   pydicom-data  https://github.com/pydicom/pydicom-data (MIT)
#   GDCM data     https://github.com/malaterre/gdcmdata (GDCM BSD-3-Clause licence, as quoted in pydicom's test-file README)
#   NEMA WG-04    compression test set (David Clunie, 2004), mirrored in https://github.com/chafey/dagcom-test-data (MIT)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p sample_data/compressed_multiframe && cd sample_data/compressed_multiframe
PYD=https://raw.githubusercontent.com/pydicom/pydicom-data/abc42b90985fb6cf385aa4af766d2c9c94a257a4/data_store/data
GDCM=https://raw.githubusercontent.com/malaterre/gdcmdata/9b38ac7a8baee4e943505ab3c56fb7d19d54754f
WG04=https://raw.githubusercontent.com/chafey/dagcom-test-data/cba25ef82382b3e5b750cdaf312124963d31362f/dicom/WG04
get() { [ -f "$1" ] || curl -L --fail --retry 3 -sS -o "$1" "$2"; }
get color3d_jpeg_baseline.dcm            "$PYD/color3d_jpeg_baseline.dcm"
get emri_small.dcm                       "$PYD/emri_small.dcm"
get emri_small_jpeg_2k_lossless.dcm      "$PYD/emri_small_jpeg_2k_lossless.dcm"
get EnhancedWithIPPPerFrameIOPShared.dcm "$GDCM/EnhancedWithIPPPerFrameIOPShared.dcm"
get MR1_J2KI                             "$WG04/compsamples_j2k/IMAGES/J2KI/MR1_J2KI"
get MR3_J2KI                             "$WG04/compsamples_j2k/IMAGES/J2KI/MR3_J2KI"
sha256sum -c --quiet <<'SUMS'
c8798b8abf8ae0a18e8c9952e7c7f75f3cc8465234b1b63f9e3ba3bebb9d5625  color3d_jpeg_baseline.dcm
151233ec63f64ebb63b979df51aa827cd612a53422c073f6ef341770c7bc9a56  emri_small.dcm
b2b4063359a08ed3b0afa9f4e4f72f84af79e5116515b446d9a30da9dc7f1888  emri_small_jpeg_2k_lossless.dcm
535679f02f157f78072ad8cb5eec20705db616924b6634bca67c28e419789cc8  EnhancedWithIPPPerFrameIOPShared.dcm
ee43a6b22c478a5707515119449cc2a03e43376066468cc1ff9cbc3cc215d7e4  MR1_J2KI
f9c5cee35a52494d060176a1fddb287445a9822e18fb554982dd42afd3044a10  MR3_J2KI
SUMS
