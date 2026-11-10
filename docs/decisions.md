# Decisions

Short log of decisions that shape BodyGlass. Newest first. Add an entry in the PR that makes or changes one.

## 2026-11 Verify cached AI model sources before reuse
Every model file encountered during download or resume is checked against the selected pinned manifest,
even when it already exists. LFS files use SHA-256; small non-LFS files use the Git blob hash. A mismatch
stops setup, removes the rejected file and leaves the model unavailable; the user can retry to download
that file from the pinned revision. Valid cached files still avoid a download. This applies to original
MLX sources, community prebuilt files and GGUF files before conversion or marking setup ready.
This does not revalidate already-installed models on every inference or authenticate locally converted
MLX output. It adds no dependencies or network requests beyond the existing user-started download flow.

## 2026-11 FastSurfer code is commit-pinned and verified before extraction
The installer fetches FastSurfer v2.6.0 by its resolved commit and checks a committed SHA-256 digest before
extracting it. Existing installs without the verification marker are replaced on the next setup run; checkpoints
are preserved and still checked by the existing weight-download loop. Extraction and layout checks happen in a
fresh staging directory before replacing installed code. The marker records installation provenance, not ongoing
local-file integrity. If GitHub regenerates the archive, setup fails closed: review the upstream commit and archive
before deliberately updating the digest. Do not bypass the check or automatically trust a new digest.

## 2026-11 DICOM series metadata renders as text
The scan summary, plain-language series list, technical details and neck-add notification escape
DICOM-derived strings before building HTML. Series identifiers are also escaped in quoted attributes,
while the rebuild callback receives the original identifier through the browser's decoded dataset.
Descriptions, exclusion reasons and technical metadata remain readable text, never page markup.
This uses the existing HTML-escaping helper and adds no dependencies or runtime network calls.

## 2026-11 AI conversations and attachments belong to one case
Changing or closing a case clears the active conversation, drafts, attachments and preview, and stops its answer.
Each request saves only to its original case and conversation. Late stream, file and model-loading callbacks are
ignored after navigation; browser-storage failures cannot prevent cleanup. View attachments are available only
once the selected case has finished rendering. Existing saved chats remain in their original per-case storage.

## 2026-11 CI privacy guard blocks scans, meshes, weights and secrets
A `privacy-guard` CI job checks every tracked file against the Privacy, Model weights and Secrets sections of
`.gitignore` (read from the file, so the lists can't drift; names compared ignoring case, as on macOS) and against DICOM, gzip and NIfTI headers, because
`git add -f` or an unexpected file name bypasses `.gitignore`. Public samples are allowed only when listed in
`tests/public_samples.txt` with a source URL and licence. If it finds a real file, a person decides what to do;
the guard never deletes files or rewrites history.

## 2026-11 Compressed DICOM tests use public samples and GDCM as a test-only reference
JPEG and JPEG 2000 multi-frame decoding is tested with public files fetched by `tools/fetch_compressed_samples.sh`
(pinned commits, SHA-256 checked, never committed). Frames decoded by pylibjpeg, which the app ships, are compared
with an uncompressed copy or with GDCM, installed in CI only. Samples come only from the official pydicom
and GDCM test-data repositories. GDCM is pinned by version and hash in `requirements-test.txt`. Lossy files must match
within 1 % of the stored range, except lossy colour JPEG at 1.5 %, because JPEG decoders legitimately upsample chroma
differently by a few levels. No public multi-frame JPEG 2000 lossy file exists, so the test packs two real single-frame
lossy CT codestreams into one file. Files in a compression no
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
