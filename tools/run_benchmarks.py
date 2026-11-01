"""Run the pipeline on every public test dataset; print method, timings and volumes.
Use an empty output folder: cases that already exist there are reused, not re-timed."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bodyglass import pipeline  # noqa: E402

DATASETS = {
    "siemens_mprage_0 (test_study, 11 series)": "sample_data/test_study",
    "datalad studyforrest T1w": "sample_data/example-dicom-structural",
    "OpenNeuro ds000001 sub-01": "sample_data/dicom_ds000001",
    "OpenNeuro ds003097 sub-0001": "sample_data/dicom_ds003097",
    "OpenNeuro ds000105 sub-1": "sample_data/dicom_ds000105",
}
method = os.environ.get("BODYGLASS_STRIP", "synthstrip")
out_root = sys.argv[1] if len(sys.argv) > 1 else f"/workspace/bodyglass/bench_{method}"
rows = []
for name, folder in DATASETS.items():
    cid = pipeline.run(folder, out_root)
    m = json.load(open(os.path.join(out_root, cid, "manifest.json")))
    rows.append({"dataset": name, "caseId": cid, "picked": m["series"]["chosen"]["description"],
                 "method": m["brainExtraction"]["method"], **m["volumesMl"],
                 "t_strip": m["timing"].get("synthstrip"), "t_total": m["timing"]["total"]})
    print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open(os.path.join(out_root, "benchmark.json"), "w"), indent=1)
