"""docs/bench.md's Arm-against-x86 tables, and the macro split README.md, docs/report.md and bench/README.md quote,
are the committed JSON in results/, row for row. `python -m tests.test_bench_doc` prints the rows."""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARM, X86 = "linux-aarch64", "linux-x86_64"
RUN = "micro_run39220769299.json"  # this run's aarch64 micro; results/linux-aarch64/micro.json stays the cost table
SECTION = (REPO / "docs" / "bench.md").read_text().split("## Arm against x86 on GitHub\n")[1].split("\n## ")[0]
LOOP_METRICS = (("jobs_per_s",), ("job_ms", "p50"), ("job_ms", "p95"), ("step_ms", "p50"), ("step_ms", "p95"),
                ("cpu_s_per_job",))
STAGES = {"kleidicv_covered_imgproc": "pyramid, saliency, warp, quality gate", "segmentation": "classical segmenter",
          "measure_thinning": "crack thinning", "decode": "decode", "python_overhead": "Python outside the timed ops"}


def load(arm: str, name: str) -> dict:
    return json.loads((REPO / "results" / arm / name).read_text())


def ms(v: float) -> str:
    return f"{v:.2f}" if v < 10 else f"{v:.1f}"


def kernel_rows():
    a, x = ({(r["kernel"], r["size"], r["threads"]): r["p50_ms"] for r in load(arm, "kernels.json")["results"]}
            for arm in (ARM, X86))
    for k, s, t in a:
        if t == 1:
            yield f"| {k} | {s} | " + " | ".join(f"{ms(a[k, s, n])} | {ms(x[k, s, n])} | {x[k, s, n] / a[k, s, n]:.2f}"
                                                 for n in (1, 4)) + " |"


def dnn_rows():
    a, x = ({r["threads"]: r for r in load(arm, "kernels.json")["dnn"]} for arm in (ARM, X86))
    for t, r in a.items():
        yield (f"| NanoDet-Plus | {r['size']} | {t} | {ms(r['p50_ms'])} / {ms(r['p95_ms'])} | "
               f"{ms(x[t]['p50_ms'])} / {ms(x[t]['p95_ms'])} | {x[t]['p50_ms'] / r['p50_ms']:.2f} |")


def loop(arm: str) -> dict:
    return {(r["threads"], r["variant"]): r for r in load(arm, "loop.json")["results"]}


def get(r: dict, path: tuple) -> float:
    for p in path:
        r = r[p]
    return r


def loop_rows():
    a, x = loop(ARM), loop(X86)
    pair = lambda r, k: f"{r[k]['p50']:.1f} / {r[k]['p95']:.1f}"
    for t, v in a:
        ra, rx = a[t, v], x[t, v]
        yield (f"| {t} | {v} | {ra['jobs_per_s']:.3f} | {rx['jobs_per_s']:.3f} | {pair(ra, 'job_ms')} | "
               f"{pair(rx, 'job_ms')} | {pair(ra, 'step_ms')} | {pair(rx, 'step_ms')} | {ra['cpu_s_per_job']:.3f} | "
               f"{rx['cpu_s_per_job']:.3f} |")


def null_rows():
    for name, arm in (("Arm", ARM), ("x86", X86)):
        res = loop(arm)
        for t in sorted({t for t, _ in res}):
            ch = lambda v, m: f"{100 * (get(res[t, v], m) / get(res[t, 'off'], m) - 1):+.1f} %"  # noqa: B023
            yield f"| {name}, {t} | " + " | ".join(f"{ch('warp+segment', m)} ({ch('warp', m)})" for m in LOOP_METRICS) + " |"


def macro_rows():
    (a, sa), (x, sx) = ((d["amdahl"], d["summary"]) for d in (load(ARM, "macro.json"), load(X86, "macro.json")))
    cell = lambda d, k: f"{d['ms_per_image'][k]:.1f} ({100 * d['fraction'][k]:.1f} %)"
    for k, label in STAGES.items():
        yield f"| {label} | {cell(a, k)} | {cell(x, k)} | {x['ms_per_image'][k] / a['ms_per_image'][k]:.2f} |"
    pct = lambda s: " / ".join(f"{s['per_image_ms'][q]:.1f}" for q in ("p50", "p95", "p99"))
    yield f"| whole image, p50 / p95 / p99 | {pct(sa)} | {pct(sx)} | {sx['per_image_ms']['p50'] / sa['per_image_ms']['p50']:.2f} |"


def job_ops_rows(size="2048x1536"):
    t, a, x = ({r["op"]: r["p50_ms"] for r in d["job_ops"]["results"] if r["size"] == size}
               for d in (load(ARM, "micro.json"), load(ARM, RUN), load(X86, "micro.json")))
    for op in t:
        yield f"| `{op}` | {t[op]:.3f} | {a[op]:.3f} | {x[op]:.3f} | {x[op] / a[op]:.2f} |"


ROWS = (kernel_rows, dnn_rows, loop_rows, null_rows, macro_rows, job_ops_rows)


def test_arm_x86_tables_are_the_json():
    rows = [r for f in ROWS for r in f()]
    assert len(rows) == 24 + 2 + 9 + 6 + 6 + 7
    assert [r for r in rows if r not in SECTION] == []


def test_arm_x86_provenance_is_the_json():
    files = [(ARM, f) for f in (RUN, "macro.json", "kernels.json", "loop.json")]
    files += [(X86, f) for f in ("micro.json", "macro.json", "kernels.json", "loop.json")]
    for arm, f in files:
        doc = load(arm, f)
        p = doc["provenance"]
        for k in ("cpu", "build_info_sha256", "cv2_custom_hal", "cv2_baseline", "cv2_dispatch", "python", "numpy"):
            assert p[k] in SECTION, (arm, f, k)
        assert p["cv2"] == "5.0.0" and p["logical_cpus"] == 4 and p["build"] == "stock (not COOL)"
    for arm, (lo, hi) in ((ARM, ("0.68", "1.41")), (X86, ("1.02", "1.62"))):
        docs = [load(arm, f) for a, f in files if a == arm]
        provs = [d["provenance"] for d in docs] + [d[k]["provenance"] for d in docs for k in ("job_ops", "dnn") if "job_ops" in d]
        loads = [p["load_avg"][0] for p in provs] + [d["load_avg_end"][0] for d in docs]
        assert (f"{min(loads):.2f}", f"{max(loads):.2f}") == (lo, hi)
        assert f"{lo} to {hi}" in SECTION


def test_macro_split_quotes_are_the_json():
    m = load(ARM, "macro.json")
    f, n = m["amdahl"]["fraction"], m["summary"]["images"]
    pct = {k: round(100 * v) for k, v in f.items()}
    want = [f"{n}-image macro", f"{pct['kleidicv_covered_imgproc']} % of a job",
            f"{pct['segmentation']} % in the classical segmenter", f"{pct['measure_thinning']} % in crack thinning",
            f"{pct['decode']} % in decode", f"{pct['python_overhead']} % in Python",
            f"{m['amdahl']['end_to_end_speedup_if_covered_ops_get']['1.8x']:.2f}×"]
    for path in ("README.md", "docs/report.md", "bench/README.md"):
        text = " ".join((REPO / path).read_text().split())
        assert [w for w in want if w not in text] == [], path


def test_arm_x86_summary_is_the_json():
    """The few Arm-against-x86 figures README.md and docs/report.md repeat from the tables."""
    la, lx = loop(ARM), loop(X86)
    jobs = sorted(la[t, "off"]["jobs_per_s"] / lx[t, "off"]["jobs_per_s"] for t in (1, 2, 4))
    sa, sx = (load(arm, "macro.json")["summary"] for arm in (ARM, X86))
    ka, kx = ({(r["kernel"], r["size"], r["threads"]): r["p50_ms"] for r in load(arm, "kernels.json")["results"]}
              for arm in (ARM, X86))
    wins = {t: sum(kx[k] > ka[k] for k in ka if k[2] == t) for t in (1, 4)}
    da, dx = ({r["threads"]: r["p50_ms"] for r in load(arm, "kernels.json")["dnn"]} for arm in (ARM, X86))
    want = [f"{jobs[0]:.2f} to {jobs[-1]:.2f} times", f"{sa['images_per_s'] / sx['images_per_s']:.2f} times",
            f"{sa['per_image_ms']['p50']} against {sx['per_image_ms']['p50']}", f"{wins[1]} of the 24",
            f"{wins[4]} of 24", f"{dx[1] / da[1]:.2f} of Arm's time"]
    assert wins == {1: 12, 4: 23}
    assert [w for w in want[3:] if w not in " ".join(SECTION.split())] == []
    for path in ("README.md", "docs/report.md"):
        text = " ".join((REPO / path).read_text().split())
        assert [w for w in want if w not in text] == [], path


if __name__ == "__main__":
    for f in ROWS:
        print(f.__name__, *f(), sep="\n")
