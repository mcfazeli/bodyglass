# Decisions

Short log of decisions that shape BodyGlass. Newest first. Add an entry in the PR that makes or changes one.

## 2026-11 Compressed DICOM tests use public samples and GDCM as a test-only reference
JPEG and JPEG 2000 multi-frame decoding is tested with public files fetched by `tools/fetch_compressed_samples.sh`
(pinned commits, SHA-256 checked, never committed). Frames decoded by pylibjpeg, which the app ships, are compared
with an uncompressed copy or with GDCM, installed in CI only. No public multi-frame JPEG 2000 lossy file exists,
so the test packs two real single-frame NEMA WG-04 lossy codestreams into one file. Files in a compression no
installed decoder can read are listed but not used, with a message naming the compression.

## 2026-10 Enhanced multi-frame MR DICOM is supported
Enhanced (multi-frame) MR files are split into stacks by their per-frame geometry and scored like classic series.

## 2026-10 AI chat is optional and private by default
Ask AI is off until the user sets it up in the app. The default runs a vision-capable model locally with MLX;
ChatGPT is optional, via Sign in with ChatGPT (OAuth with PKCE) from the local server, never the web page.
Shared context shows as removable tags. Answers never give a diagnosis.

## 2026-10 Neck exam is aligned to the head, or refused
TotalSegmentator labels C1 to T1, discs and cord; the neck is rigidly aligned on the upper vertebrae seen in both
exams and refused, with an explanation, if it doesn't fit within 3 mm.

## Founding decisions
- Everything runs on the user's Mac; scans never leave it. Network use is limited to setup, model downloads the
  user starts, and ChatGPT if they choose it.
- Series are picked from DICOM headers, never folder names.
- Educational only: no statements that anything is normal or abnormal.
- Only public or synthetic data in the repo, tests and screenshots; no faces.
