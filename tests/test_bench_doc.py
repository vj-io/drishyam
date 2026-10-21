"""docs/bench.md's Arm-against-x86 tables and the figures its text takes from them, and the macro split and the
Arm-against-x86 summary README.md, docs/report.md and bench/README.md quote, are the committed JSON in results/.
`python -m tests.test_bench_doc` prints the rows."""
import json
import re
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARM, X86, MAC = "linux-aarch64", "linux-x86_64", "local-macos-x86"
RUN = "micro_run39220769299.json"  # this run's aarch64 micro; results/linux-aarch64/micro.json stays the cost table
FILES = {ARM: (RUN, "macro.json", "kernels.json", "loop.json"),
         X86: ("micro.json", "macro.json", "kernels.json", "loop.json")}
BENCH = (REPO / "docs" / "bench.md").read_text()
SECTION = BENCH.split("## Arm against x86 on GitHub\n")[1].split("\n## ")[0]
LOOP_METRICS = (("jobs_per_s",), ("job_ms", "p50"), ("job_ms", "p95"), ("step_ms", "p50"), ("step_ms", "p95"),
                ("cpu_s_per_job",))
STAGES = {"kleidicv_covered_imgproc": "pyramid, saliency, warp, quality gate", "segmentation": "classical segmenter",
          "measure_thinning": "crack thinning", "decode": "decode", "python_overhead": "Python outside the timed ops"}
PROV = ("label", "cpu", "logical_cpus", "machine", "platform", "cv2", "build", "build_info_sha256", "cv2_baseline",
        "cv2_dispatch", "cv2_custom_hal", "cv2_parallel_framework", "python", "numpy")


def load(arm: str, name: str) -> dict:
    return json.loads((REPO / "results" / arm / name).read_text())


def flat(text: str) -> str:
    return " ".join(text.split())


def ms(v: float) -> str:
    return f"{v:.2f}" if v < 10 else f"{v:.1f}"


def ands(v: list) -> str:
    return ", ".join(v[:-1]) + " and " + v[-1]


def blocks(d: dict) -> list:
    """A file's provenance and its micro sub-blocks' (job_ops and DNN have their own)."""
    return [d["provenance"]] + [d[k]["provenance"] for k in ("job_ops", "dnn") if "job_ops" in d]


def prov(arm: str) -> tuple[dict, list]:
    """The provenance all four files of the run share, and every 1-minute load reading in them."""
    docs = [load(arm, f) for f in FILES[arm]]
    ps = [{k: d["provenance"][k] for k in PROV} for d in docs]
    assert all(p == ps[0] for p in ps), arm
    return ps[0], [p["load_avg"][0] for d in docs for p in blocks(d)] + [d["load_avg_end"][0] for d in docs]


def kernels(arm: str) -> dict:
    return {(r["kernel"], r["size"], r["threads"]): r["p50_ms"] for r in load(arm, "kernels.json")["results"]}


def provenance_rows():
    (a, la), (x, lx) = prov(ARM), prov(X86)
    cells = {"runner (`provenance.label`)": lambda p: p["label"],
             "CPU (`provenance.cpu`), logical CPUs": lambda p: f"{p['cpu']}, {p['logical_cpus']}",
             "OS": lambda p: f"{p['machine'].rsplit(' ', 1)[0]}, glibc {p['platform'].split('-with-glibc')[1]}",
             "OpenCV": lambda p: f"{p['cv2']}, {p['build']}",
             "`build_info_sha256`": lambda p: f"`{p['build_info_sha256']}`",
             "Baseline; dispatched": lambda p: f"{p['cv2_baseline']}; {p['cv2_dispatch']}",
             "Custom HAL": lambda p: p["cv2_custom_hal"], "parallel framework": lambda p: p["cv2_parallel_framework"],
             "Python; NumPy": lambda p: f"{p['python']}; {p['numpy']}"}
    for label, f in cells.items():
        yield f"| {label} | {f(a)} | {'the same' if label == 'OS' and f(x) == f(a) else f(x)} |"
    yield f"| 1-minute load average, lowest to highest | {min(la):.2f} to {max(la):.2f} | {min(lx):.2f} to {max(lx):.2f} |"


def kernel_rows():
    a, x = kernels(ARM), kernels(X86)
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


def change(res: dict, t: int, m: tuple, v: str = "warp+segment") -> float:
    """Variant v against prefetch off, in %, on one loop metric."""
    return 100 * (get(res[t, v], m) / get(res[t, "off"], m) - 1)


def pc(v: float) -> str:
    return f"{v:+.1f} %"


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
            ch = lambda v, m: pc(change(res, t, m, v))  # noqa: B023
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


ROWS = (provenance_rows, kernel_rows, dnn_rows, loop_rows, null_rows, macro_rows, job_ops_rows)


def test_arm_x86_tables_are_the_json():
    rows = [r for f in ROWS for r in f()]
    assert len(rows) == 10 + 24 + 2 + 9 + 6 + 6 + 7
    assert [r for r in rows if r not in SECTION] == []


def test_arm_x86_prose_is_the_json():
    """Every figure the section's text takes from this run's files, and the direction each of its comparisons states."""
    (pa, la), (px, lx) = prov(ARM), prov(X86)
    ka, kx = kernels(ARM), kernels(X86)
    sizes = sorted({s for _, s, _ in ka}, key=lambda s: int(s.split("x")[0]))
    pairs = [(k, s) for k, s, t in ka if t == 1]
    one, four = ({p: kx[(*p, t)] / ka[(*p, t)] for p in pairs} for t in (1, 4))  # x86 ÷ Arm: above 1 Arm is faster
    r = lambda k: [one[k, s] for s in sizes]
    span = lambda v: f"{min(v):.2f} to {max(v):.2f}"
    spa, spx = ({p: d[(*p, 1)] / d[(*p, 4)] for p in pairs} for d in (ka, kx))  # 1-to-4-thread speed-up
    bh, cn, st, pyr, blob, cc, cl, wg = (r(k) for k in ("blackhat_tophat", "canny", "scharr_tensor", "pyramid", "blob",
                                                        "components_contours", "clahe_lab", "warp_gaze"))
    assert max(bh + cn + st + blob[1:] + pyr[2:]) < 1 < min(cc + cl + wg + blob[:1] + pyr[:2])
    assert all(0.45 < v < 0.55 for v in bh) and all(1.8 < v < 2.2 for v in cc) and f"{blob[1]:.2f}" == f"{blob[2]:.2f}"
    close = [v for v in one.values() if abs(v - 1) < 0.06]
    win4 = {p: v for p, v in four.items() if v >= 1}
    lo4, hi4 = min(win4, key=win4.get), max(win4, key=win4.get)
    bh1, st1 = ("blackhat_tophat", "1024x768"), ("scharr_tensor", "1024x768")
    assert four.keys() - win4.keys() == {bh1} and (lo4, hi4) == (("blob", "2048x1536"), ("clahe_lab", "2048x1536"))
    assert abs(spa[bh1] - 1) < 0.02 and abs(spx[bh1] - 1) < 0.02
    assert [(n, p) for n, d in (("Arm", spa), ("x86", spx)) for p, v in d.items() if v < 1] == [("Arm", st1), ("x86", st1)]
    above = sorted(p for p, v in spa.items() if v > pa["logical_cpus"])  # faster than the thread count allows
    assert above == [("pyramid", "4000x3000"), ("scharr_tensor", "2048x1536")]
    pyr_a, pyr_x = [spa["pyramid", s] for s in sizes], {f"{spx['pyramid', s]:.1f}" for s in sizes}
    assert len(pyr_x) == 1
    da, dx = ({r["threads"]: r["p50_ms"] for r in load(arm, "kernels.json")["dnn"]} for arm in (ARM, X86))
    kc = [load(arm, "kernels.json")["config"] for arm in (ARM, X86)]
    assert kc[0] == kc[1]

    lpa, lpx = loop(ARM), loop(X86)
    rows = [*lpa.values(), *lpx.values()]
    jobs, steps = rows[0]["jobs"], rows[0]["steps"]
    assert all((r["jobs"], r["steps"], r["decisions_identical"]) == (jobs, steps, True) for r in rows)
    assert all(load(arm, "loop.json")["decisions_identical_across_threads"] for arm in (ARM, X86))
    pf = lambda res: {(r["prefetch"]["warp_hit"], r["prefetch"]["dropped"], r["prefetch"]["submitted"])
                      for r in res.values() if r["variant"] != "off"}
    ((hit, drop, sub),) = pf(lpa) | pf(lpx)
    mac = loop(MAC)
    assert pf(mac) == {(2 * hit, 2 * drop, 2 * sub)} and load(MAC, "loop.json")["config"]["rounds"] == 2
    lc = load(ARM, "loop.json")["config"]
    assert lc == load(X86, "loop.json")["config"] and lc["rounds"] == 1
    demo = sum(p.startswith("bridge_") for p in lc["photos"])
    c = lambda res, t, m, v="warp+segment": pc(change(res, t, m, v))
    J, P50, P95, S50, S95, CPU = LOOP_METRICS
    null = max(abs(change(res, t, m, "warp")) for res in (lpa, lpx) for t, _ in res for m in LOOP_METRICS)
    steps1 = max(abs(change(res, 1, m)) for res in (lpa, lpx) for m in (S50, S95))
    assert all(40 < change(res, 1, CPU) < 60 for res in (lpa, lpx))  # "half again the CPU"
    jps = lambda res, t: res[t, "off"]["jobs_per_s"]
    cpu = lambda res, t: res[t, "off"]["cpu_s_per_job"]
    assert jps(lpx, 4) <= jps(lpx, 2)  # "x86 gained nothing from 2 to 4 threads"

    ma, mx = load(ARM, "macro.json"), load(X86, "macro.json")
    sa, sx = ma["summary"], mx["summary"]
    ia, ix = ({p["path"]: p for p in m["per_image"]} for m in (ma, mx))
    odd = [k for k in ia if (ia[k]["steps"], ia[k]["pixels_frac"]) != (ix[k]["steps"], ix[k]["pixels_frac"])]
    assert ia.keys() == ix.keys() and len(odd) == 1
    mean = lambda d, f: sum(p[f] for p in d.values()) / len(d)
    wa, wx = (m["amdahl"]["ms_per_image"] for m in (ma, mx))
    fa = ma["amdahl"]["fraction"]
    gap = {k: wx[k] - wa[k] for k in wa}
    assert wx["decode"] < wa["decode"]
    e2e = lambda m: m["amdahl"]["end_to_end_speedup_if_covered_ops_get"]["1.8x"]
    assert all(m["config"]["workers"] == m["config"]["cv2_threads_per_worker"] == 1 for m in (ma, mx))
    assert sa["pixel_cap"] == sx["pixel_cap"] and "2048x1536" in sa["dataset"]

    table, run = (load(ARM, f) for f in ("micro.json", RUN))
    t, a = ({(r["op"], r["size"]): r["p50_ms"] for r in d["job_ops"]["results"]} for d in (table, run))
    below = {k: 100 * (1 - a[k] / t[k]) for k in t}
    dk = ("downscale", "1024x768")
    rest = [v for k, v in below.items() if k != dk]
    assert min(below.values()) > 0 and max(below, key=below.get) == dk
    xo = {r["op"]: r["p50_ms"] for r in load(X86, "micro.json")["job_ops"]["results"] if r["size"] == "2048x1536"}
    ao = {r["op"]: r["p50_ms"] for r in run["job_ops"]["results"] if r["size"] == "2048x1536"}
    xa = {op: xo[op] / ao[op] for op in xo}
    assert min(xa["segment:accurate"], xa["segment:fast"], xa["downscale"]) > 1 > max(xa["saliency"], xa["quality"])
    m0, m1 = ({(r["op"], r["size"]): r["p50_ms"] for r in d["results"]} for d in (table, run))
    moved = {k: 100 * (m1[k] / m0[k] - 1) for k in m0}
    big = sorted(k for k, v in moved.items() if abs(v) >= 3.5)
    assert big == [("pyrDown", "4000x3000"), ("resize_area", "4000x3000")] and all(moved[k] < 0 for k in big)
    assert table["provenance"]["build_info_sha256"] == pa["build_info_sha256"]
    day = lambda d: date.fromisoformat(d["provenance"]["timestamp"][:10])
    assert (day(run) - day(table)).days == 1
    ts = sorted(p["timestamp"] for arm in FILES for f in FILES[arm] for p in blocks(load(arm, f)))
    assert ts[0][:10] == ts[-1][:10]
    for arm, loads in ((ARM, la), (X86, lx)):  # "the highest readings came after the 4-thread kernels"
        assert max(loads) == load(arm, "kernels.json")["load_avg_end"][0]
    assert pa["logical_cpus"] == px["logical_cpus"]
    wheel = re.search(r'name = "opencv-python-headless"\nversion = "([^"]+)"', (REPO / "uv.lock").read_text())[1]

    want = [f"on {ts[0][:10]}: the files' start timestamps run from {ts[0][11:19]} to {ts[-1][11:19]} UTC",
            (f"Of its {len(moved)} op timings (p50), {len(moved) - 2} moved by less than 3.5 %, and `pyrDown` and the "
             f"`INTER_AREA` resize at 4000x3000 ran {-moved[big[0]]:.1f} % and {-moved[big[1]]:.1f} % faster in this run"),
            f"The same wheel version ({wheel}, `uv.lock`)", f"and {pa['logical_cpus']} logical CPUs each",
            f"timed on {day(table)}", f"Their Python is {pa['python']}", f"about 1 on {pa['logical_cpus']} logical CPUs",
            f"the highest readings came after the {kc[0]['all_threads']}-thread kernels",
            f"Metric B at {ands([str(n) for n in lc['threads']])} threads",
            f"{kc[0]['repeats']} timed repeats after {kc[0]['warmup']} warm-ups; `all` is {kc[0]['all_threads']} threads",
            (f"each runner wins {len(pairs) // 2} of the {len(pairs)} kernel and size pairs, but {len(close)} of the 24 "
             f"differ by less than 6 % ({sum(v < 1 for v in close)} of x86's wins and {sum(v > 1 for v in close)} of Arm's)"),
            (f"at every size, where Arm takes about twice as long ({span(bh)}), Canny ({span(cn)}) and the structure "
             f"tensor ({span(st)}) at every size, the blob from the two larger frames ({blob[1]:.2f}) and the 4000x3000 "
             f"pyramid ({pyr[2]:.2f}). Arm wins contours and fitting ({span(cc)}), CLAHE in LAB ({span(cl)}), the oblique "
             f"gaze warp ({span(wg)}), the blob from 1024x768 ({blob[0]:.2f}) and the two smaller pyramids "
             f"({pyr[1]:.2f} and {pyr[0]:.2f}). On 4 threads Arm wins {len(win4)} of 24, by {four[lo4]:.2f} (the blob at "
             f"2048x1536) to {four[hi4]:.2f} (CLAHE at 2048x1536); the exception is the black-hat and top-hat pair at "
             f"1024x768 ({four[bh1]:.2f})"),
            f"({spa[bh1]:.2f} and {spx[bh1]:.2f} times its 1-thread speed)",
            (f"The structure tensor at {st1[1]} is the one kernel that gets slower on 4 threads, {spx[st1]:.2f} times its "
             f"1-thread speed on x86 and {spa[st1]:.2f} on Arm, so its {four[st1]:.2f} is an x86 loss"),
            (f"from 1 to 4 threads Arm's pyramid gets {min(pyr_a):.1f} to {max(pyr_a):.1f} times faster and x86's "
             f"{pyr_x.pop()} times, and no x86 kernel more than {max(spx.values()):.2f} times against up to "
             f"{max(spa.values()):.2f} on Arm"),
            (f"Arm's {spa['scharr_tensor', '2048x1536']:.2f} (the structure tensor at 2048x1536) and "
             f"{spa['pyramid', '4000x3000']:.2f} (the pyramid at 4000x3000) are above {pa['logical_cpus']} on "
             f"{pa['logical_cpus']} CPUs"),
            (f"on 4 threads Arm is {dx[4] / da[4]:.2f} times faster (Arm {da[1] / da[4]:.1f} times its one-thread speed, "
             f"x86 {dx[1] / dx[4]:.1f} times)"),
            (f"the {demo} demo photos and {len(lc['photos']) - demo} synthetic ones, one round, at "
             f"{ands([str(n) for n in lc['threads']])} threads"),
            (f"in {steps} steps over the {jobs} photos; each prefetching variant served {hit} of {steps} looks from the "
             f"cache ({hit / steps:.2f}) and dropped {drop} of its {sub} speculative looks unread"),
            f"The null arm stayed within {null:.1f} % everywhere",
            (f"jobs/s {c(lpa, 1, J)} on Arm and {c(lpx, 1, J)} on x86, job p95 {c(lpa, 1, P95)} and {c(lpx, 1, P95)}, CPU per "
             f"job {c(lpa, 1, CPU)} and {c(lpx, 1, CPU)}, steps within {steps1:.1f} %. Arm's job p50 rose "
             f"{change(lpa, 1, P50):.1f} % (null {c(lpa, 1, P50, 'warp')})"),
            (f"**x86 on 2 and 4 threads: a loss at the step.** step p95 {c(lpx, 2, S95)} and {c(lpx, 4, S95)}, step "
             f"p50 {c(lpx, 2, S50)} and {c(lpx, 4, S50)}, job p50 {c(lpx, 2, P50)} and {c(lpx, 4, P50)}, for jobs/s "
             f"{c(lpx, 2, J)} and {c(lpx, 4, J)} and CPU {c(lpx, 2, CPU)} and {c(lpx, 4, CPU)}"),
            (f"**Arm: the gain holds on 2 threads; on 4 the median pays.** On 2, jobs/s {c(lpa, 2, J)}, job p95 "
             f"{c(lpa, 2, P95)}, step p95 {c(lpa, 2, S95)} (null {c(lpa, 2, S95, 'warp')}), CPU {c(lpa, 2, CPU)}. On 4, "
             f"jobs/s {c(lpa, 4, J)} and job p95 {c(lpa, 4, P95)}, but job p50 {c(lpa, 4, P50)} and step p50 "
             f"{c(lpa, 4, S50)} (null {c(lpa, 4, P50, 'warp')} and {c(lpa, 4, S50, 'warp')})"),
            (f"Arm ran {ands([f'{jps(lpa, t) / jps(lpx, t):.2f}' for t in (1, 2, 4)])} times x86's jobs/s at 1, 2 and 4 "
             f"threads, and x86 used {ands([f'{cpu(lpx, t) / cpu(lpa, t):.2f}' for t in (1, 2, 4)])} times Arm's CPU per "
             f"job. x86 gained nothing from 2 to 4 threads ({jps(lpx, 2):.3f} to {jps(lpx, 4):.3f} jobs/s) while its CPU "
             f"per job rose from {cpu(lpx, 2):.3f} to {cpu(lpx, 4):.3f} s; Arm went from {jps(lpa, 2):.3f} to "
             f"{jps(lpa, 4):.3f} jobs/s"),
            (f"on {sa['images']} synthetic 2048x1536 images at the {100 * sa['pixel_cap']:.0f} % cap, one worker, one "
             f"OpenCV thread"),
            (f"Arm did {sa['images_per_s']:.2f} images/s and x86 {sx['images_per_s']:.2f} (Arm "
             f"{sa['images_per_s'] / sx['images_per_s']:.2f} times)"),
            (f"all but one of the {len(ia)} images, which took {ia[odd[0]]['steps']} steps on Arm and {ix[odd[0]]['steps']} "
             f"on x86: mean steps {mean(ia, 'steps'):.3f} and {mean(ix, 'steps'):.3f}, mean pixel fraction "
             f"{mean(ia, 'pixels_frac'):.4f} and {mean(ix, 'pixels_frac'):.4f}"),
            (f"Of the {sum(gap.values()):.0f} ms per image x86 needs beyond Arm, {gap['segmentation']:.0f} are the classical "
             f"segmenter and {gap['measure_thinning']:.0f} crack thinning (`measure`'s hit-or-miss thinning, "
             f"{wx['measure_thinning'] / wa['measure_thinning']:.2f} times Arm's); x86 decodes faster "
             f"({wx['decode'] / wa['decode']:.2f})"),
            (f"If the covered stages ran 1.8 times faster, a whole job would get {e2e(ma):.3f} times faster on Arm and "
             f"{e2e(mx):.3f} times on x86"),
            (f"now {round(100 * fa['segmentation'])} %; the same {round(100 * fa['kleidicv_covered_imgproc'])} % covered "
             f"and {e2e(ma):.2f} times"), f"against {sa['images_per_s']:.2f} images/s does not compare",
            f"(GitHub's arm64 runner, {day(table)}, the same build hash)",
            f"At 2048x1536, the size of the {sa['images']}-image sets",
            (f"All {len(t)} of its `job_ops` p50s are below the table's: by {min(rest):.1f} to {max(rest):.1f} %, and by "
             f"{below[dk]:.1f} % for `downscale` at 1024x768 ({a[dk]:.3f} against {t[dk]:.3f} ms)"),
            (f"segment op {xa['segment:accurate']:.2f} (`accurate`) and {xa['segment:fast']:.2f} (`fast`) times Arm's and "
             f"the downscale baseline's `INTER_AREA` resize {xa['downscale']:.2f} times, but saliency and the quality gate "
             f"{xa['saliency']:.2f} and {xa['quality']:.2f} times")]
    assert [w for w in want if w not in flat(SECTION)] == []
    bullet = (f"run 39220769299 is one run per runner and one round of Metric B. The only repeat is the aarch64 micro, "
              f"a day apart on the same build ({len(moved) - 2} of {len(moved)} ops within 3.5 %, the other two "
              f"{-moved[big[0]]:.1f} % and {-moved[big[1]]:.1f} % apart); x86 was never repeated")
    assert bullet in flat(BENCH)


def test_macro_split_quotes_are_the_json():
    m = load(ARM, "macro.json")
    f, n = m["amdahl"]["fraction"], m["summary"]["images"]
    pct = {k: round(100 * v) for k, v in f.items()}
    want = [f"{n}-image macro", f"{pct['kleidicv_covered_imgproc']} % of a job",
            f"{pct['segmentation']} % in the classical segmenter", f"{pct['measure_thinning']} % in crack thinning",
            f"{pct['decode']} % in decode", f"{pct['python_overhead']} % in Python",
            f"{m['amdahl']['end_to_end_speedup_if_covered_ops_get']['1.8x']:.2f}×"]
    for path in ("README.md", "docs/report.md", "bench/README.md"):
        text = flat((REPO / path).read_text())
        assert [w for w in want if w not in text] == [], path


def test_arm_x86_summary_is_the_json():
    """The Arm-against-x86 figures README.md and docs/report.md repeat, and the direction of each."""
    la, lx = loop(ARM), loop(X86)
    jobs = sorted(la[t, "off"]["jobs_per_s"] / lx[t, "off"]["jobs_per_s"] for t in (1, 2, 4))
    sa, sx = (load(arm, "macro.json")["summary"] for arm in (ARM, X86))
    wa, wx = (load(arm, "macro.json")["amdahl"]["ms_per_image"] for arm in (ARM, X86))
    ka, kx = kernels(ARM), kernels(X86)
    wins = {t: sum(kx[k] > ka[k] for k in ka if k[2] == t) for t in (1, 4)}
    ratio = lambda k: [kx[k, s, 1] / ka[k, s, 1] for s in ("1024x768", "2048x1536", "4000x3000")]
    close = sum(abs(kx[k] / ka[k] - 1) < 0.06 for k in ka if k[2] == 1)
    da, dx = ({r["threads"]: r["p50_ms"] for r in load(arm, "kernels.json")["dnn"]} for arm in (ARM, X86))
    assert wins == {1: 12, 4: 23}
    assert all(0.45 < v < 0.55 for v in ratio("blackhat_tophat")) and all(1.8 < v < 2.2 for v in ratio("components_contours"))
    assert wx["segmentation"] - wa["segmentation"] > (sum(wx.values()) - sum(wa.values())) / 2
    (pa, _), (px, _) = prov(ARM), prov(X86)
    assert "ipp" in px["cv2_custom_hal"] and pa["logical_cpus"] == px["logical_cpus"] and pa["python"] == px["python"]
    want = [f"{jobs[0]:.2f} to {jobs[-1]:.2f} times", f"{sa['images_per_s'] / sx['images_per_s']:.2f} times",
            f"{sa['per_image_ms']['p50']} against {sx['per_image_ms']['p50']}", f"{wins[1]} of the 24",
            f"{wins[4]} of 24", f"{dx[1] / da[1]:.2f} of Arm's time", f"{close} of the 24 differ by less than 6 %",
            "most of the gap in the classical segmenter", f"on {pa['logical_cpus']} threads Arm won {wins[4]} of 24",
            f"({px['cpu'].split(' 64-Core')[0]}, IPP HAL)", f"{pa['logical_cpus']} logical CPUs"]
    assert [w for w in want[3:7] if w not in flat(SECTION)] == []
    direction = {"README.md": "black-hat and top-hat by about 2 times",
                 "docs/report.md": "x86 about twice as fast at black-hat and top-hat, Arm about twice as fast at contours "
                                   f"and fitting), but {close} of the 24 differ by less than 6 %"}
    assert f"Python {pa['python'][:4]} on both" in flat((REPO / "docs/report.md").read_text())
    for path, phrase in direction.items():
        text = flat((REPO / path).read_text())
        assert [w for w in [*want, phrase] if w not in text] == [], path


if __name__ == "__main__":
    for f in ROWS:
        print(f.__name__, *f(), sep="\n")
