"""Fold three benchmark arms into one report: COOL gain (A over B), hardware gain (B over C), what a buyer sees (A over C).

    python -m bench.compare_arms results/A results/B results/C [--out docs/cool_benchmark.md]
                                 [--summary public/results/cool.json]

Each directory is what `bench/run_all.sh <arm>` wrote (micro.json, macro.json, parity.npz), in the order
A = COOL AMI on c8g.2xlarge, B = stock wheel on c8g.2xlarge, C = stock wheel on c7i.2xlarge. Arms that ran
different OpenCV major versions, or timed or segmented different images, are refused rather than compared.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from bench.parity import compare

ROLES = {"A": "COOL build, Graviton", "B": "stock wheel, Graviton", "C": "stock wheel, x86"}
COOL_FEE = 0.02  # $/h Marketplace fee on c8g.2xlarge (docs/PRD.md cost ledger); arm A's usd_per_hour includes it
EXPECTED = {"op_level": [1.2, 1.8], "end_to_end": [1.2, 1.3]}  # docs/PRD.md, "Expectations to state up front"


def load(d: str) -> dict:
    p = Path(d)
    with np.load(p / "parity.npz") as z:
        parity = {"crops": z["crops"], "labels": z["labels"], "meta": json.loads(str(z["meta"]))}
    return {"dir": str(p), "micro": json.loads((p / "micro.json").read_text()),
            "macro": json.loads((p / "macro.json").read_text()), "parity": parity}


def image_set(arm: dict) -> dict:
    # ponytail: macro images are matched by file name, not bytes; the runbook ships one dataset to every arm
    s = arm["macro"]["summary"]
    return {"dataset": s["dataset"], "pixel cap": s["pixel_cap"],
            "macro images": sorted(r["path"] for r in arm["macro"]["per_image"]),
            "micro op/size grid": sorted((r["op"], r["size"]) for r in arm["micro"]["results"])}


def check(arms: dict) -> None:
    """Raise ValueError unless every arm ran one OpenCV major and timed and segmented the same images as arm A."""
    majors = {}
    for role, a in arms.items():
        provs = (a["micro"]["provenance"], a["macro"]["provenance"], a["parity"]["meta"])
        if len({p["build_info_sha256"] for p in provs}) > 1:
            raise ValueError(f"arm {role} ({a['dir']}): micro, macro and parity ran different cv2 builds")
        majors[role] = provs[0]["cv2"].split(".")[0]
    if len(set(majors.values())) > 1:
        raise ValueError("OpenCV major versions differ: " + ", ".join(f"{r} {v}.x" for r, v in majors.items()))
    want = image_set(arms["A"])
    for role, a in arms.items():
        got = image_set(a)
        if bad := [k for k in want if got[k] != want[k]]:
            raise ValueError(f"arm {role} ({a['dir']}) timed different images from arm A: {', '.join(bad)} differ")
        if not np.array_equal(a["parity"]["crops"], arms["A"]["parity"]["crops"]):
            raise ValueError(f"arm {role} ({a['dir']}) segmented different parity crops from arm A; rerun its parity "
                             "step with PARITY_INPUTS pointing at the shared parity.npz")


def ratio(num: float, den: float) -> float:
    return round(num / den, 3)


def build(dirs: list[str]) -> dict:
    arms = dict(zip(ROLES, map(load, dirs), strict=True))
    check(arms)
    micro = {r: {(x["op"], x["size"]): x for x in a["micro"]["results"]} for r, a in arms.items()}
    ops = []
    for key in micro["A"]:
        t = {r: micro[r][key] for r in ROLES}
        ops.append({"op": key[0], "size": key[1],
                    **{f"{r}_{q}_ms": t[r][f"{q}_ms"] for r in ROLES for q in ("p50", "p95")},
                    "cool_gain": ratio(t["B"]["p50_ms"], t["A"]["p50_ms"]),
                    "hardware_gain": ratio(t["C"]["p50_ms"], t["B"]["p50_ms"])})
    out = {}
    for r, a in arms.items():
        s, prov, fee = a["macro"]["summary"], a["macro"]["provenance"], COOL_FEE if r == "A" else 0.0
        out[r] = {"what": ROLES[r], "dir": a["dir"], "timestamp": prov["timestamp"],
                  "instance_type": prov.get("instance_type", "not recorded"), "ami_id": prov.get("ami_id", "not recorded"),
                  "cv2": prov["cv2"], "build_info_sha256": prov["build_info_sha256"],
                  "cv2_from_opt_cool": prov["under_opt_cool"], "cv2_file": prov["cv2_file"], "cpu": prov["cpu"],
                  "python": prov["python"], "numpy": prov["numpy"],
                  "micro_cv2_threads": a["micro"]["config"]["cv2_threads"],
                  "images": s["images"], "images_per_s": s["images_per_s"], "per_image_ms": s["per_image_ms"],
                  "usd_per_hour": s["usd_per_hour"], "instance_usd_per_hour": round(s["usd_per_hour"] - fee, 4),
                  "cool_fee_usd_per_hour": fee,
                  "usd_per_1k_images": round(s["usd_per_hour"] / 3600 / s["images_per_s"] * 1000, 6),
                  "amdahl": a["macro"]["amdahl"]["fraction"], "amdahl_ms": a["macro"]["amdahl"]["ms_per_image"]}
    ips = {r: out[r]["images_per_s"] for r in ROLES}
    gm = lambda k: round(statistics.geometric_mean(o[k] for o in ops), 3)
    return {"dirs": list(dirs), "expected": EXPECTED, "arms": out, "ops": ops,
            "op_level_geomean": {"cool_gain": gm("cool_gain"), "hardware_gain": gm("hardware_gain")},
            "end_to_end": {"cool_gain": ratio(ips["A"], ips["B"]), "hardware_gain": ratio(ips["B"], ips["C"]),
                           "a_over_c": ratio(ips["A"], ips["C"])},
            "parity_a_vs_b": compare(arms["A"]["parity"]["labels"], arms["B"]["parity"]["labels"])}


def table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head),
            *("| " + " | ".join(map(str, row)) + " |" for row in rows), ""]


def markdown(rep: dict) -> str:
    arms, e2e, gm, par = rep["arms"], rep["end_to_end"], rep["op_level_geomean"], rep["parity_a_vs_b"]
    span = lambda k: "–".join(map(str, rep["expected"][k])) + "×"
    parts = list(arms["A"]["amdahl"])
    return "\n".join([
        "# COOL on Graviton: three-arm benchmark", "",
        (f"Generated by `python -m bench.compare_arms {' '.join(rep['dirs'])}` from the JSON and `parity.npz` files in "
         "those directories. Git keeps only the JSON, so re-running it needs the run's own `parity.npz` files. "
         "Speed-up of X over Y is Y's p50 time over X's for single ops and X's images/s over Y's for whole jobs. "
         "**COOL gain** is A over B (same hardware, different OpenCV build); **hardware gain** is B over C (the same "
         "stock wheel version, built for each architecture, so it includes the differences between the two builds)."),
        "",
        "## Arms", "",
        *table(["Arm", "What", "Instance", "AMI", "cv2", "Build-info sha256", "cv2 from /opt/cool", "CPU",
                "cv2 threads (micro)"],
               [[r, a["what"], a["instance_type"], a["ami_id"], a["cv2"], f"`{a['build_info_sha256'][:16]}`",
                 "yes" if a["cv2_from_opt_cool"] else "no", a["cpu"], a["micro_cv2_threads"]] for r, a in arms.items()]),
        "## Whole job (macro, one worker)", "",
        (f"Expected (docs/PRD.md): {span('end_to_end')} end to end. Measured: COOL gain {e2e['cool_gain']}×, "
         f"hardware gain {e2e['hardware_gain']}×, A over C {e2e['a_over_c']}×."), "",
        *table(["Arm", "Images", "Images/s", "Per-image p50 / p95 ms", "Instance $/h", "COOL fee $/h", "$/1k images"],
               [[r, a["images"], a["images_per_s"], f"{a['per_image_ms']['p50']} / {a['per_image_ms']['p95']}",
                 a["instance_usd_per_hour"], a["cool_fee_usd_per_hour"], a["usd_per_1k_images"]]
                for r, a in arms.items()]),
        "$/1k images = (instance + COOL fee) $/h × hours ÷ images, billing the whole instance for one worker.", "",
        "## Single ops (micro, ms)", "",
        (f"Expected (docs/PRD.md): {span('op_level')} op level on KleidiCV ops. Measured geometric mean over the rows "
         f"below: COOL gain {gm['cool_gain']}×, hardware gain {gm['hardware_gain']}×."), "",
        *table(["Op", "Size", "A p50", "A p95", "B p50", "B p95", "C p50", "C p95", "COOL gain (A over B)",
                "Hardware gain (B over C)"],
               [[o["op"], o["size"], *(o[f"{r}_{q}_ms"] for r in ROLES for q in ("p50", "p95")),
                 f"{o['cool_gain']}×", f"{o['hardware_gain']}×"] for o in rep["ops"]]),
        "## Where a job's time goes (Amdahl split: share of wall time, ms per image)", "",
        *table(["Part", *ROLES], [[p, *(f"{100 * arms[r]['amdahl'][p]:.1f} % ({arms[r]['amdahl_ms'][p]} ms)"
                                       for r in ROLES)] for p in parts]),
        "## Mask parity, A vs B", "",
        (f"{par['crops']} crops, identical inputs in every arm: pixel agreement {par['pixel_agreement']}, worst crop "
         f"{par['worst_crop_agreement']}, {par['identical_crops']} of {par['crops']} crops identical."), "",
        *table(["Class", "IoU"], [[c, "absent in both" if v is None else v] for c, v in par["class_iou"].items()]),
        "## Provenance", "",
        *(f"- **{r}** (`{a['dir']}`, {a['timestamp']}): cv2 {a['cv2']} from `{a['cv2_file']}`, Python {a['python']}, "
          f"NumPy {a['numpy']}, build-info sha256 `{a['build_info_sha256']}`" for r, a in arms.items()),
        ""])


def summary(rep: dict) -> dict:
    """The few numbers the website shows."""
    keep = ("what", "instance_type", "cv2", "cv2_from_opt_cool", "images_per_s", "usd_per_hour", "usd_per_1k_images")
    return {"expected": rep["expected"], "end_to_end": rep["end_to_end"], "op_level_geomean": rep["op_level_geomean"],
            "parity_a_vs_b": {k: rep["parity_a_vs_b"][k] for k in ("crops", "pixel_agreement", "worst_crop_agreement")},
            "arms": {r: {**{k: a[k] for k in keep}, "covered_fraction": a["amdahl"]["kleidicv_covered_imgproc"]}
                     for r, a in rep["arms"].items()}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m bench.compare_arms", description=__doc__.split("\n\n")[0])
    ap.add_argument("arms", nargs=3, metavar="DIR", help="results directories, in the order A (COOL/c8g), "
                                                         "B (stock/c8g), C (stock/c7i)")
    ap.add_argument("--out", default="docs/cool_benchmark.md", help="markdown report")
    ap.add_argument("--summary", default="public/results/cool.json", help="small JSON for the website")
    a = ap.parse_args(argv)
    try:
        rep = build(a.arms)
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"refusing to compare: {exc}")
    Path(a.out).write_text(markdown(rep))
    Path(a.summary).write_text(json.dumps(summary(rep), indent=1) + "\n")
    e = rep["end_to_end"]
    print(f"wrote {a.out} and {a.summary}: end to end COOL gain {e['cool_gain']}x, hardware gain {e['hardware_gain']}x")


if __name__ == "__main__":
    main()
