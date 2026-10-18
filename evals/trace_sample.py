"""The report's recorded trace: one job on a bundled sample at a 30 % pixel cap, on the published fixed arm64 costs.

    python -m evals.trace_sample > docs/trace-bridge_03.ndjson

Deterministic (fixed costs on a virtual clock, fixed job id). Steps keep the fields section 10 quotes; findings,
review requests and the done summary are kept without images, the kill token or the signed work order.
"""

import json
from pathlib import Path

from drishyam import imaging, privacy
from drishyam.agent import JobConfig
from drishyam.segment import default_segmenter
from evals.harness import FixedCostJob, cost_rows, op_costs


def main() -> None:
    img = imaging.decode_image(Path("public/samples/bridge_03.jpg").read_bytes())
    privacy.privacy_pass(img)  # as the CLI and the web app do before the job
    costs = op_costs(cost_rows("linux-aarch64", default_segmenter().name), img.shape[1], img.shape[0])
    job = FixedCostJob(img, JobConfig(pixel_cap=0.3, thumbs=False), costs, job_id="j_sample")
    for ev in job.run():
        if ev["type"] == "step":
            obs = ev.get("observation", {})
            ev = {"type": "step", "step": ev["step"], "decision": ev["decision"], "cause": ev["cause"], "rationale": ev["rationale"],
                  "observation": {k: obs[k] for k in ("entropy", "mask_area_px", "classes") if k in obs}}
        elif ev["type"] == "finding":
            ev = {**ev, "finding": {k: v for k, v in ev["finding"].items() if k != "crop_jpeg_b64"}}
        elif ev["type"] == "done":
            ev = {k: ev[k] for k in ("type", "status", "cause", "summary", "unvisited")}
        elif ev["type"] != "review_request":
            continue
        print(json.dumps(ev))


if __name__ == "__main__":
    main()
