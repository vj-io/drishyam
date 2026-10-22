"""docs/report.md's classical-engine tables, its failure gallery and its Arm-against-x86 table, and the table and
numbers the results-at-a-glance blocks of README.md and docs/report.md quote, are the committed JSON in
public/results/ and results/ (the Lambda timings: docs/aws.md's smoke record). `python -m tests.test_report_doc`
prints the rows."""
import json
import re
from pathlib import Path

from tests.test_bench_doc import ARM, X86, kernels, loop
from tests.test_bench_doc import load as bench

REPO = Path(__file__).resolve().parents[1]
REPORT, README, AWS = ((REPO / f).read_text() for f in ("docs/report.md", "README.md", "docs/aws.md"))
CAPS = ("0.05", "0.1", "0.2", "0.3", "0.5", "1.0")  # evals.harness.BUDGETS: the order of policies["drishyam"]
SETS = {"synth": "summary.json", "large": "large/summary.json", "codebrim": "codebrim/summary.json"}
CLASSES = ("crack", "spalling", "efflorescence", "exposed_rebar", "corrosion")
VARIANTS = {"no-uncertainty": "no uncertainty term", "no-seed": "no seeding from looks", "no-floor": "no stop floor",
            "no-survey": "no survey look", "no-verify": "no verify step", "governor-on": "governor on"}


def load(path: str) -> dict:
    return json.loads((REPO / path).read_text())


def flat(text: str) -> str:
    return " ".join(text.split())


def ci(c: list) -> str:
    return f"{c[0]:.2f}–{c[1]:.2f}"


def summary(name: str) -> dict:
    return load(f"public/results/{SETS[name]}")


def drishyam(d: dict, cap: str) -> dict:
    return d["policies"]["drishyam"][CAPS.index(cap)]


def policies(d: dict, caps: tuple, uncapped: str) -> list:
    """(label, row): one downscale, the three grids (coarsest first), then Drishyam at each cap in `caps`."""
    p = d["policies"]
    rows = [("downscale once (1024 px)", p["downscale"][0])]
    rows += list(zip(("tiling, L2 grid", "tiling, L1 grid", "tiling, L0 (exhaustive, 20 % overlap)"), p["tiling"], strict=True))
    return rows + [(f"**Drishyam**, {uncapped if c == '1.0' else f'cap {round(float(c) * 100)} %'}", drishyam(d, c)) for c in caps]


def recall(label: str, r: dict) -> str:
    b = "**" if "Drishyam" in label else ""
    if r["recall_ci"] == [0.0, 0.0]:
        return f"{b}{r['recall']:.2f}{b} (no 512² look fits after the overview on a 3 MP image)"
    return f"{b}{r['recall']:.2f}{b} ({ci(r['recall_ci'])})"


def synth_rows(name: str, caps: tuple = ("0.3", "1.0"), uncapped: str = "uncapped", appendix: bool = False):
    for label, r in policies(summary(name), caps, uncapped):
        extra = (f" {'–' if r['precision'] is None else format(r['precision'], '.3f')} | {r['fp_per_image']:.1f} | "
                 f"{'–' if r['hits_per_matched_gt'] is None else format(r['hits_per_matched_gt'], '.2f')} | {r['usd']:.6f} |"
                 if appendix else f" {r['fp_per_image']:.1f} |")
        yield (f"| {label} | {r['pixels_frac']:.2f} | {recall(label, r)} | {r['recall_iou_only']:.2f} | "
               f"{r['chance_recall']:.2f} ({r['chance_recall_iou_only']:.2f}) |" + extra)


def codebrim_rows(caps: tuple = ("0.3", "1.0"), uncapped: str = "uncapped"):
    for label, r in policies(summary("codebrim"), caps, uncapped):
        yield (f"| {label} | {r['pixels_frac']:.2f} | {recall(label, r)} | {r['recall_iou_only']:.2f} "
               f"({ci(r['recall_iou_only_ci'])}) | {r['chance_recall']:.2f} ({r['chance_recall_iou_only']:.2f}) | "
               f"{r['boxes_per_image']:,.0f} |")


def class_rows(name: str):
    pc = summary(name)["per_class"]
    for label, key in (("Drishyam @30 %", "drishyam@0.3"), ("tiling L0", "tiling@1.0"), ("downscale", "downscale")):
        yield f"| {label} | " + " | ".join(f"{pc[key][c]:.2f}" for c in CLASSES) + " |"


def ablation_rows():
    ds = load("public/results/ablations.json")["datasets"]
    cap = {k: v["by_cap"]["0.3"] for k, v in ds.items()}
    pooled = {k: re.search(r"seeds pooled per image: (\d\.\d\d)-(\d\.\d\d)", v["random_order"][0]).groups()
              for k, v in ds.items()}
    yield "| full agent (recall) | " + " | ".join(f"{c['full']['recall']:.2f}" for c in cap.values()) + " |"
    yield "| random gaze order, seeds 0–2 (pooled CI only) | " + " | ".join(f"{a}–{b}" for a, b in pooled.values()) + " |"
    for k, label in VARIANTS.items():
        yield f"| {label} | " + " | ".join(f"{c[k]['ratio_vs_full']:.2f} ({ci(c[k]['ratio_vs_full_ci'])})" for c in cap.values()) + " |"


def failure_rows():
    cases, seen = load("public/results/failures.json"), {}
    for c in cases:
        assert "|" not in c["what_happened"] + c["rule_that_would_catch_it"]
        cat = list(dict.fromkeys(x["category"] for x in cases)).index(c["category"]) + 1
        seen[cat] = seen.get(cat, -1) + 1
        yield (f"| ![FIG. 5.{cat}({'ab'[seen[cat]]})](../public/results/{c['image']}) | {c['category']} | "
               f"{c['what_happened']} | {c['rule_that_would_catch_it']} |")


def arm_rows():
    la, lx = loop(ARM), loop(X86)
    ka, kx = kernels(ARM), kernels(X86)
    (ma, mx) = (bench(arm, "macro.json")["summary"] for arm in (ARM, X86))
    for t in (1, 4):
        a = sum(kx[k] > ka[k] for k in ka if k[2] == t)
        yield f"| OpenCV kernels won, {t} thread{'s' * (t > 1)} (of 24 kernel and size pairs) | {a} | {24 - a} | |"
    for t in (1, 4):
        a, x = la[t, "off"]["jobs_per_s"], lx[t, "off"]["jobs_per_s"]
        yield f"| agent loop, jobs/s, {t} thread{'s' * (t > 1)} | {a:.2f} | {x:.2f} | {a / x:.2f} |"
    yield (f"| macro, {ma['images']} synthetic images, 1 thread: images/s | {ma['images_per_s']:.2f} | "
           f"{mx['images_per_s']:.2f} | {ma['images_per_s'] / mx['images_per_s']:.2f} |")
    yield f"| macro, per-image p50 ms | {ma['per_image_ms']['p50']} | {mx['per_image_ms']['p50']} | |"


ROWS = {"synthetic": lambda: synth_rows("synth"), "large": lambda: synth_rows("large"), "codebrim": codebrim_rows,
        "appendix synthetic": lambda: synth_rows("synth", CAPS, "cap 100 %", appendix=True),
        "appendix synthetic classes": lambda: class_rows("synth"),
        "appendix large": lambda: synth_rows("large", CAPS, "cap 100 %"),
        "appendix codebrim": lambda: codebrim_rows(CAPS, "cap 100 %"), "appendix codebrim classes": lambda: class_rows("codebrim"),
        "ablations": ablation_rows, "failures": failure_rows, "arm": arm_rows}


def test_report_tables_are_the_json():
    rows = {k: list(f()) for k, f in ROWS.items()}
    assert {k: len(v) for k, v in rows.items()} == {
        "synthetic": 6, "large": 6, "codebrim": 6, "appendix synthetic": 10, "appendix synthetic classes": 3,
        "appendix large": 10, "appendix codebrim": 10, "appendix codebrim classes": 3, "ablations": 8, "failures": 12, "arm": 6}
    assert {k: [r for r in v if r not in REPORT] for k, v in rows.items()} == {k: [] for k in rows}


def halts(name: str, cap: str) -> dict:
    return summary(name)["agent_metrics"]["by_budget"][cap]["halt_causes"]


def classical_quotes() -> list:
    """The classical engine's figures the report's prose and both glance blocks repeat."""
    s, lg, cb = (summary(n) for n in SETS)
    hs, hl, hc = (d["headline_stats"] for d in (s, lg, cb))
    p = lambda d, k: d["policies"]["tiling"][("L2", "L1", "L0").index(k)]
    return [(f"{hs['recall_ratio_vs_tiling']:.2f}× exhaustive tiling ({ci(hs['recall_ratio_vs_tiling_ci'])}) at "
            f"{hs['pixels_ratio_vs_tiling']:.2f}× its pixels"),
            f"{hs['recall_ratio_vs_downscale']:.2f}× one 1024-px downscale ({ci(hs['recall_ratio_vs_downscale_ci'])})",
            (f"{drishyam(cb, '0.3')['recall']:.2f} of CODEBRIM's defects, {hc['recall_ratio_vs_downscale']:.2f}× one downscale "
            f"({ci(hc['recall_ratio_vs_downscale_ci'])})"),
            f"{hl['recall_ratio_vs_downscale']:.2f}× ({ci(hl['recall_ratio_vs_downscale_ci'])})",
            (f"L2 tiling grid finds {p(lg, 'L2')['recall']:.2f} with {p(lg, 'L2')['pixels_frac']:.2f} of the pixels against "
            f"Drishyam's {drishyam(lg, '0.3')['pixels_frac']:.2f}"),
            f"ends {halts('synth', '0.3')['cap-hit']} of 200 synthetic and {halts('codebrim', '0.3')['cap-hit']} of 200 CODEBRIM jobs",
            f"the stop rule ends {halts('synth', '1.0')['stop-rule']} of 200 synthetic jobs"]


GLANCE = (("v1", "v1 (first look policy)"), ("v1-policy2", "**v1-policy2 (live)**"), ("v2a", "v2a"),
          ("v2b", "v2b"), ("v2c", "v2c"), ("classical-v1", "classical engine"))
INSPECTION_PHOTOS = 2536  # one detailed drone inspection of one bridge (docs/impact-sources.md, ODOT SPR 787)


def backlog_quotes() -> list:
    """The one measured backlog of an inspection's size (results/backlog_2536/summary.json): what it was made of, what
    it queued for a person and what it left not assessed, as the glance blocks, the README and section 10 say it."""
    b = load("results/backlog_2536/summary.json")
    c, synth = b["counts"], b["by_split"]["synthetic damage"] + b["by_split"]["synthetic clean"]
    assert b["photos"] == INSPECTION_PHOTOS and c["critical"] == b["criticals_queued"] and c["not assessed"] == b["cleared"] + b["not_reached"]
    return [(f"the one measured backlog of {b['photos']:,} photos, {synth:,} of them synthetic, queued {b['criticals_queued']} "
             f"critical findings from the {b['run']} photos it ran and left {c['not assessed']:,} not assessed"),
            (f"A {b['photos']:,}-photo backlog ({synth:,} synthetic photos and the {b['by_split']['codebrim dev100']} CODEBRIM dev "
             f"photos) ran end to end"),
            (f"{b['cleared']} cleared, {b['run']} run, {b['criticals_queued']} critical findings queued for a person, and "
             f"{c['not assessed']:,} photos listed as not assessed"),
            (f"backlog (`results/backlog_2536/summary.md`; {synth:,} synthetic photos and the {b['by_split']['codebrim dev100']} "
             f"CODEBRIM dev photos) it cleared {b['cleared']} photos"),
            f"and listed {c['not assessed']:,} photos as not assessed (cleared or not reached)"]


def gate_quote() -> str:
    """The line under both glance tables: the human gate in the unit a person signs (critical findings per photo, by each
    finding's final measure on both sides, as section 8's gate table), beside the box count the table shows."""
    _, cb, _, _ = p2()
    m, mv = cb["matrix"]["v1-policy2@0.3"], cb["matrix"]["v1@0.3"]
    return (f"A person signs critical findings, not boxes: about {mv['signoffs_per_image']:.2f} per CODEBRIM photo under v1's "
            f"first policy and {m['signoffs_per_image']:.2f} under v1-policy2, the live one")


def glance_rows():
    """The held-out table both glance blocks open with: one row per segmenter at the 30 % cap."""
    out = load("results/heldout_models.json")
    t, n = out["table"], out["datasets"]["clean"]["n"]
    for m, label in GLANCE:
        c, s, k = (t[f"{d}@0.3"][m] for d in ("codebrim", "synth", "clean"))
        yield (f"| {label} | {c['recall']:.2f} ({ci(c['recall_ci'])}) | {c['chance_recall']:.2f} | "
               f"{c['recall_iou_only']:.2f} ({c['chance_recall_iou_only']:.2f}) | {s['recall']:.2f} ({ci(s['recall_ci'])}) | "
               f"{round(k['images_with_any'] * n)} of {n} | {c['critical_per_image']:.2f} |")


def heldout_quotes() -> list:
    out = load("results/heldout_models.json")
    t, pr = out["table"], out["pairs"]
    s1, sc = t["synth@0.3"]["v1"], t["synth@0.3"]["classical-v1"]
    r, rs = pr["codebrim@0.3: v1 / classical-v1"]["recall"], pr["synth@0.3: v1 / classical-v1"]["recall"]
    ms = pr["codebrim@0.3: v1 / classical-v1"]["ms"]
    crit = [t["codebrim@0.3"][m]["critical_per_image"] for m in ("v1", "classical-v1")]
    assert out["dev_choice"]["confirms"] is True and out["where_each_loses"]["v1"] == []
    assert t["clean@0.3"]["v1"]["boxes_per_image"] == 0  # "puts no box on a clean photo"
    return [f"v1 finds {r[0]:.2f}× ({ci(r[1])}) its recall on CODEBRIM and {rs[0]:.2f}× ({ci(rs[1])}) on synth200",
            f"where the classical engine puts {t['clean@0.3']['classical-v1']['boxes_per_image']:.2f} per photo",
            f"it sends {crit[0] / crit[1]:.1f} times the critical boxes to the human gate (boxes, not the findings a person signs; "
            + backlog_quotes()[0] + ")",
            f"its modelled ms per CODEBRIM job are {ms[0]:.2f}× (CI {ci(ms[1])})",
            (f"no efflorescence ({s1['per_class']['efflorescence']:.2f}) and almost no exposed rebar "
             f"({s1['per_class']['exposed_rebar']:.2f}), where the classical engine finds {sc['per_class']['efflorescence']:.2f} "
             f"and {sc['per_class']['exposed_rebar']:.2f}")]


def p2() -> tuple:
    """v1-policy2's held-out block, its CODEBRIM and synth200 reads, and its pair with v1."""
    out = load("results/heldout_models.json")
    b = out["v1-policy2"]
    assert b["sets"] == ["codebrim", "synth", "clean"] and all(b["dev_rule_on_held_out"]["clauses"].values())
    assert all(all(r["baselines_same_as_v1s_run"].values()) for r in b["read"].values())
    return b, b["read"]["codebrim"], b["read"]["synth"], out["pairs"]


def policy2_quotes() -> list:
    """What both glance blocks and report section 8 say of v1-policy2's one held-out read, losses with gains."""
    _, cb, _, pr = p2()
    m, mv, m1 = cb["matrix"]["v1-policy2@0.3"], cb["matrix"]["v1@0.3"], cb["matrix"]["v1-policy2@0.1"]
    crit, rec = pr["codebrim@0.3: v1-policy2 / v1"]["critical"], pr["codebrim@0.3: v1-policy2 / v1"]["recall"]
    srec, cr = pr["synth@0.3: v1-policy2 / v1"]["recall"], m["vs v1@0.3"]["crit_recall"]
    pc, pv = cb["rows"]["v1-policy2@0.3"]["per_class"], cb["rows"]["v1@0.3"]["per_class"]
    rt, rv = cb["rates"]["v1-policy2@0.3"], cb["rates"]["v1@0.3"]
    t2, t21 = m["vs tiling@L2"], m1["vs tiling@L2"]
    return [(f"a person signs about {m['signoffs_per_image']:.2f} critical findings per photo instead of "
             f"{mv['signoffs_per_image']:.2f}, and the critical boxes fall to {m['critical_per_image']:.2f} per photo from "
             f"{mv['critical_per_image']:.2f} ({crit[0]:.2f}×, CI {ci(crit[1])})"),
            f"lists {m['needs_look_per_image']:.2f} masks per photo as needs-a-look",
            f"recall is unchanged ({rec[0]:.2f}×, CI {ci(rec[1])}), on synth200 {srec[0]:.2f}× ({ci(srec[1])})",
            (f"critical-class recall is not higher ({cr[0]:.2f}×, {ci(cr[1])}; exposed rebar "
             f"{pc['exposed_rebar']:.2f} against {pv['exposed_rebar']:.2f})"),
            (f"verify now fires on {100 * rt['verify_rate']:.1f} % of looks ({100 * rt['confirm_rate']:.1f} % confirm "
             f"looks) instead of {100 * rv['verify_rate']:.1f} %, escalation on {100 * rt['escalation_rate']:.1f} %"),
            (f"{t2['recall_shown'][0]:.2f}× its recall at {t2['pixels']:.2f}× its pixels, and "
             f"{t21['recall_shown'][0]:.2f}× ({ci(t21['recall_shown'][1])}) at the 10 % cap")]


def policy2_cost_quotes() -> list:
    """Report section 8's losses for v1-policy2: synth200's boxes, the hard subsets' small critical boxes, ms, pixels."""
    _, cb, sy, pr = p2()
    d = pr["synth@0.3: v1-policy2 / v1"]["boxes_per_photo_diff"]
    hs, hc, m = sy["hard"]["rows"], cb["hard"]["rows"], cb["matrix"]["v1-policy2@0.3"]
    ms = pr["codebrim@0.3: v1-policy2 / v1"]["ms"]
    return [f"more boxes per photo ({d[0]:+.2f}, CI {d[1][0]:+.2f} to {d[1][1]:+.2f})",
            (f"it finds {hs['v1-policy2@0.3']['small_critical_recall']:.2f} of synth200's small critical boxes against "
             f"v1's {hs['v1@0.3']['small_critical_recall']:.2f}, and {hc['v1-policy2@0.3']['small_critical_recall']:.2f} "
             f"against {hc['v1@0.3']['small_critical_recall']:.2f} on CODEBRIM, where L1 tiling finds "
             f"{hc['tiling@L1']['small_critical_recall']:.2f}"),
            (f"modelled ms per CODEBRIM job are {ms[0]:.2f}× v1's (CI {ci(ms[1])}), pixels "
             f"{m['vs v1@0.3']['pixels']:.2f}×"),
            f"v1-policy2's exact count is {m['signoffs_exact_per_image']:.2f}"]


def policy2_rows() -> list:
    """Report section 8's v1-policy2 matrix: each policy's pixels, modelled ms, recall of what a person is shown and
    drafted critical-class recall on CODEBRIM eval200 and synth200, and its recall ratios to the baselines and v1."""
    _, cb, sy, _ = p2()
    f = lambda v, c: "–" if v is None else f"{v:.3f} ({ci(c)})"
    r = lambda m, k: f"{m[k]['recall_shown'][0]:.2f} ({ci(m[k]['recall_shown'][1])})" if k in m else "–"
    rows = []
    for name, rd in (("CODEBRIM eval200", cb), ("synth200", sy)):
        for k in ("downscale", "tiling@L2", "tiling@L1", "v1@0.3", "v1-policy2@0.3", "v1@0.1", "v1-policy2@0.1"):
            m = rd["matrix"][k]
            rows.append(f"| {name} | {k} | {m['pixels_frac']:.3f} | {m['ms_per_job']:.0f} | "
                        f"{f(m['recall_shown'], m['recall_shown_ci'])} | {f(m['crit_recall'], m['crit_recall_ci'])} | "
                        f"{m['critical_per_image']:.2f} | {r(m, 'vs tiling@L2')} | {r(m, 'vs tiling@L1')} | "
                        f"{r(m, 'vs downscale')} | {r(m, 'vs v1@' + k.rpartition('@')[2]) if k.startswith('v1-') else '–'} |")
    return rows


def policy2_gate_rows() -> list:
    """The human gate and the hard subsets, v1 against v1-policy2 at the 30 % cap."""
    b, cb, sy, _ = p2()
    rows = []
    for name, rd in (("CODEBRIM eval200", cb), ("synth200", sy), ("clean", b["read"]["clean"])):
        h = rd["hard"]
        for k in ("v1@0.3", "v1-policy2@0.3"):
            m, x, t = rd["matrix"][k], h["rows"].get(k, {}), rd["rates"][k]
            small = "–" if x.get("small_critical_recall") is None else f"{x['small_critical_recall']:.2f}"
            rows.append(f"| {name} | {k} | {m['critical_per_image']:.2f} | {m['signoffs_per_image']:.2f} | "
                        f"{m['needs_look_per_image']:.2f} | {100 * t['verify_rate']:.1f} % | {100 * t['escalation_rate']:.1f} % | "
                        f"{h['photos']} of {h['of']} | {small} |")
    return rows


def v1_look_quotes() -> list:
    """How much v1 zooms on CODEBRIM at the headline cap, and the tiling grids beside it (its held-out summaries)."""
    cb, sy = (load(f"public/results/{d}_dacl10k/summary.json") for d in ("codebrim", "synth"))
    a = cb["agent_metrics"]["by_budget"]["0.3"]
    t2, t1 = cb["policies"]["tiling"][0], sy["policies"]["tiling"][1]  # the L2 and L1 grids
    d, s = drishyam(cb, "0.3"), drishyam(sy, "0.3")
    return [(f"an escalation fired on {100 * a['escalation_rate']:.1f} % of v1's looks and a verify on "
             f"{100 * a['verify_rate']:.1f} %"),
            (f"an L2 tiling grid finds {t2['recall']:.2f} with {t2['pixels_frac']:.2f} of the pixels, against the agent's "
             f"{d['recall']:.2f} with {d['pixels_frac']:.2f}"),
            (f"an L1 grid finds {t1['recall']:.2f} with {t1['pixels_frac']:.2f} of the pixels, against {s['recall']:.2f} "
             f"with {s['pixels_frac']:.2f}")]


def pitch_quotes() -> list:
    """The README's opening paragraph: v1's CODEBRIM recall, its ratio to the classical engine and its chance score, the
    L2 tiling grid beside it, and the live policy's exposed-rebar recall on CODEBRIM and synth200."""
    t = load("results/heldout_models.json")
    c, ratio = t["table"]["codebrim@0.3"]["v1"], t["pairs"]["codebrim@0.3: v1 / classical-v1"]["recall"][0]
    l2 = t["v1-policy2"]["read"]["codebrim"]["rows"]["tiling@L2"]
    rb = [t["table"][f"{d}@0.3"]["v1-policy2"]["per_class"]["exposed_rebar"] for d in ("codebrim", "synth")]
    return [(f"finds {c['recall']:.2f} of the labelled defects, {ratio:.1f} times what the classical engine finds, but "
             f"boxes placed at random would already find {c['chance_recall']:.2f}, and a plain L2 tiling grid finds "
             f"{l2['recall']:.2f}"),
            f"It rarely finds exposed rebar: {rb[0]:.2f} of it on these photos and {rb[1]:.2f} on synthetic ones"]


def live_quotes() -> list:
    """What the glance blocks take from docs/aws.md: the deployed commit and the Lambda step p95s (no JSON holds them)."""
    aws = flat(AWS)
    deployed = re.search(r"redeployed from commit (\w{7})", aws).group(1)
    v1 = re.search(r"with v1 a step costs (\d\.\d\d) to (\d\.\d\d) s at p95", aws).groups()
    cl = re.search(r"against (\d+\.\d) to (\d+\.\d) ms for the classical engine", aws).groups()
    return [f"deployed commit {deployed}", f"its step p95 was {v1[0]} to {v1[1]} s, against {cl[0]} to {cl[1]} ms"]


def adoption_quotes() -> list:
    tune = load("results/dev/tune_sweep.json")["decision"]["not_adopted"]["tau_scout 0.7 (ts07)"]
    cas = load("results/dev/cascade_dacl10k_v2a+v2b.json")["splits"]
    i8 = load("results/local-macos-x86/int8.json")["models"]
    fw = lambda m: f"{i8[m]['forward512_threads1']['p50_ms']:.1f}"
    qdq = i8["static_qdq/new"]["vs_fp32_new"]["class_iou"]["crack"]
    c = {k: cas[k]["rows"]["cascade@0.3"] for k in ("real", "synth_dev")}
    assert not i8["dynamic/new"]["runs"] and not i8["dynamic/classic"]["runs"] and not i8["fp32/classic"]["runs"]
    shape = "!isDynamicShape in function 'parseShape'"  # section 6: the classic importer's stop on v1's decoder
    assert all(shape in i8[f"{m}/classic"]["error"] for m in ("fp32", "static_qdq", "static_qoperator")) and shape in REPORT
    assert re.search(r"x1\.03 at 0\.3 \(CI 0\.98-1\.09", tune) and re.search(r"x1\.22 at 0\.3 \(CI 1\.09-1\.43", tune)
    assert "About a third of the recall gain" in tune
    return ["synth_dev gain was not measurable (1.03×, 0.98–1.09)",
            (f"lower recall than v2a alone on CODEBRIM dev100 ({c['real']['vs_ref']:.2f}×, {ci(c['real']['vs_ref_ci'])}) and "
            f"on synth_dev ({c['synth_dev']['vs_ref']:.2f}×, {ci(c['synth_dev']['vs_ref_ci'])})"),
            f"{fw('static_qdq/new')} and {fw('static_qoperator/new')} ms against FP32's {fw('fp32/new')}",
            f"crack masks overlap FP32's at IoU {qdq:.2f}"]


def arm_quotes() -> list:
    la, lx = loop(ARM), loop(X86)
    (ma, mx) = (bench(arm, "macro.json")["summary"] for arm in (ARM, X86))
    return [f"{ma['images_per_s']:.2f} images/s against x86's {mx['images_per_s']:.2f}",
            (f"{la[1, 'off']['jobs_per_s'] / lx[1, 'off']['jobs_per_s']:.2f} times x86's jobs/s on one thread and "
            f"{la[4, 'off']['jobs_per_s'] / lx[4, 'off']['jobs_per_s']:.2f} times on four")]


def test_glance_quotes_are_the_json():
    for name, text in (("README.md", README), ("docs/report.md", REPORT)):
        glance = text.split("## Results at a glance\n")[1].split("\n## ")[0]
        assert [r for r in glance_rows() if r not in glance] == [], name
        want = classical_quotes()[:3] + heldout_quotes() + v1_look_quotes() + live_quotes() + policy2_quotes() + [gate_quote()]
        assert [w for w in want if w not in flat(glance)] == [], name
    assert [w for w in pitch_quotes() if w not in flat(README.split("## Results at a glance\n")[0])] == []
    # moved out of the glance: tuning, cascade, INT8 and prefetching to both docs' results; Arm against x86 to the
    # README's (the report has it as the "arm" table)
    assert [w for w in adoption_quotes() if w not in flat(README) or w not in flat(REPORT)] == []
    assert [w for w in arm_quotes() if w not in flat(README)] == []
    assert [w for w in classical_quotes()[5:] if w not in flat(README) or w not in flat(REPORT)] == []


def spread(rows: list, field: str, digits: int = 2) -> str:
    v = [r[field] for r in rows if r[field] is not None]
    return f"{min(v):.{digits}f} to {max(v):.{digits}f}"


def report_quotes() -> list:
    """The figures the prose of sections 3, 7 and 8 takes from the classical summaries, the ablations and v1's runs."""
    s, lg, cb = (summary(n) for n in SETS)
    hs, hl, hc = (d["headline_stats"] for d in (s, lg, cb))
    t0, t1, t2 = s["policies"]["tiling"][::-1]  # L0, L1, L2
    d3, d5, d10 = (drishyam(s, c) for c in ("0.3", "0.5", "1.0"))
    am, box, hair = s["agent_metrics"], s["agent_metrics"]["box_convention"]["recall"], s["strata"]["recall"]
    fa, pcl, pcc = am["false_accept_critical"], lg["per_class"]["tiling@1.0"], cb["per_class"]["drishyam@0.3"]
    cbr = [r for k in ("downscale", "tiling", "drishyam") for r in cb["policies"][k]]
    ab = load("public/results/ablations.json")["datasets"]["synth200"]["by_cap"]["1.0"]
    abq = {k: f"{ab[k]['ratio_vs_full']:.2f}×" for k in ab}
    abci = {k: ci(ab[k]["ratio_vs_full_ci"]) for k in ab}
    cl = load("public/results/clean/summary.json")["false_alarms"]["drishyam@0.3"]
    v1 = {k: v["v1"] for k, v in load("results/heldout_models.json")["table"].items() if "v1" in v}
    v1cb = load("public/results/codebrim_dacl10k/summary.json")["policies"]
    vq = lambda k, f, where: f"{v1[k][f]:.2f}× {where} ({ci(v1[k][f + '_ci'])})"
    return [
        (f"the stop rule ends {halts('synth', '1.0')['stop-rule']} of 200 synthetic jobs and "
        f"{halts('codebrim', '1.0')['stop-rule']} of 200 CODEBRIM jobs"),
        vq("codebrim@0.3", "vs_downscale", "one downscale on CODEBRIM"),
        vq("large@0.3", "vs_downscale", "at 4608×3456"),
        vq("synth@0.3", "vs_tiling_L0", "exhaustive tiling on the 2048×1536 set"),
        (f"finds {v1cb['tiling'][0]['recall']:.2f} with {v1cb['tiling'][0]['pixels_frac']:.2f} of the pixels against "
        f"the agent's {v1cb['drishyam'][3]['recall']:.2f} with {v1cb['drishyam'][3]['pixels_frac']:.2f}"),
        f"finds **{d3['recall']:.2f}** of the defects using **{d3['pixels_frac']:.2f}** of the image's pixels",
        (f"**{hs['recall_ratio_vs_tiling']:.2f}×** tiling's recall (95 % CI {ci(hs['recall_ratio_vs_tiling_ci'])}) "
        f"at **{hs['pixels_ratio_vs_tiling']:.2f}×** its pixels"),
        f"Drishyam is {hs['recall_ratio_vs_downscale']:.2f}× (95 % CI {ci(hs['recall_ratio_vs_downscale_ci'])})",
        f"{d3['recall'] / t1['recall']:.2f}× L1 tiling", f"{d3['recall'] / t2['recall']:.2f}× L2 tiling",
        (f"it is {hs['recall_ratio_iou_only_vs_tiling']:.2f}× L0 tiling "
        f"(95 % CI {ci(hs['recall_ratio_iou_only_vs_tiling_ci'])})"),
        f"is {box['union']['drishyam@0.3'] / box['union']['tiling@L0']:.2f}×",
        f"it is {box['as_specified']['drishyam@0.3'] / box['as_specified']['tiling@L0']:.2f}×",
        (f"{d5['recall']:.2f} at {d5['pixels_frac']:.2f} of the pixels, {d5['recall'] / t0['recall']:.2f}× tiling's "
        f"recall at {d5['pixels_frac'] / t0['pixels_frac']:.2f}× its pixels"),
        (f"(the other {halts('synth', '1.0')['exhausted']} when every candidate has been looked at), at "
        f"{d10['pixels_frac']:.2f} of the pixels with {d10['recall'] / t0['recall']:.2f}× tiling's recall"),
        (f"precision {d3['precision']:.3f} for Drishyam at 30 % ({d3['fp_per_image']:.1f} false boxes per image) "
        f"against {s['policies']['downscale'][0]['precision']:.3f}"),
        f"({d3['hits_per_matched_gt']:.2f} predictions per matched box",
        (f"exhaustive tiling finds {hair['tiling@L0']['hairline_crack']:.2f} of them, one downscale "
        f"{hair['downscale']['hairline_crack']:.2f}, Drishyam {hair['drishyam@0.3']['hairline_crack']:.2f} at the "
        f"30 % cap and {hair['drishyam@1.0']['hairline_crack']:.2f} uncapped"),
        f"Drishyam at 30 % {fa['drishyam@0.3']['rate']:.2f} against exhaustive tiling {fa['tiling@L0']['rate']:.2f}",
        f"escalation rate {round(100 * am['escalation_rate'])} %",
        f"{halts('synth', '0.3')['cap-hit']} of 200 jobs hit the cap",
        f"({halts('synth', '0.3')['stop-rule']} on the stop rule",
        (f"{drishyam(lg, '0.3')['recall']:.2f} against {lg['policies']['downscale'][0]['recall']:.2f}, "
        f"{hl['recall_ratio_vs_downscale']:.2f}× (95 % CI {ci(hl['recall_ratio_vs_downscale_ci'])})"),
        f"spalling {pcl['spalling']:.2f}, exposed rebar {pcl['exposed_rebar']:.2f}",
        f"({hl['recall_ratio_vs_tiling']:.2f}×, which formally",
        (f"the stop rule ends {halts('large', '1.0')['stop-rule']} of 100 jobs by itself (the other "
        f"{halts('large', '1.0')['exhausted']}"),
        (f"**{hc['recall_ratio_vs_tiling']:.2f}×** exhaustive tiling's recall "
        f"(95 % CI {ci(hc['recall_ratio_vs_tiling_ci'])}"),
        f"({hc['recall_ratio_iou_only_vs_tiling']:.2f}×, {ci(hc['recall_ratio_iou_only_vs_tiling_ci'])}, when a match",
        f"**{hc['recall_ratio_vs_downscale']:.2f}×** a single 1024-px downscale ({ci(hc['recall_ratio_vs_downscale_ci'])})",
        f"already score {spread(cbr, 'chance_recall')} under the lenient rule",
        f"recall is {spread(cbr, 'recall_iou_only')} and chance {spread(cbr, 'chance_recall_iou_only')}",
        f"Precision (one-to-one) is {spread(cbr, 'precision', 4)}",
        f"cracks ({pcc['crack']:.2f}) and corrosion ({pcc['corrosion']:.2f})",
        (f"the stop rule ends {halts('codebrim', '1.0')['stop-rule']} of 200 jobs, at "
        f"{drishyam(cb, '1.0')['pixels_frac']:.2f}"),
        f"Uncapped it still costs recall ({abq['no-survey']}, {abci['no-survey']}, at 2048×1536",
        (f"{abq['no-uncertainty']} without the uncertainty term ({abci['no-uncertainty']}) and {abq['no-seed']} "
        f"without the seeding ({abci['no-seed']}), at {ab['no-uncertainty']['pixels_frac']:.2f} and "
        f"{ab['no-seed']['pixels_frac']:.2f} of the pixels against the full agent's {ab['full']['pixels_frac']:.2f}"),
        (f"Uncapped there it is {abq['no-verify']} ({abci['no-verify']}) without it, at "
        f"{ab['no-verify']['pixels_frac']:.2f} of the pixels"),
        f"reports {cl['boxes_per_image']:.2f} boxes ({cl['critical_per_image']:.2f} critical)",
        f"(what a Drishyam finding is), {cl['union_per_image']:.2f}"]


def readme_quotes() -> list:
    """The classical figures and ablations README.md's Results section repeats."""
    s, lg, cb = (summary(n) for n in SETS)
    hs, hl, hc = (d["headline_stats"] for d in (s, lg, cb))
    d3, l3, c3 = (drishyam(d, "0.3") for d in (s, lg, cb))
    ds = load("public/results/ablations.json")["datasets"]
    a3 = {k: v["by_cap"]["0.3"] for k, v in ds.items()}
    a1 = ds["synth200"]["by_cap"]["1.0"]
    pooled = {k: "–".join(re.search(r"pooled per image: (\d\.\d\d)-(\d\.\d\d)", v["random_order"][0]).groups())
              for k, v in ds.items()}
    q = lambda r, k: f"{r[k]['ratio_vs_full']:.2f}×, {ci(r[k]['ratio_vs_full_ci'])}"
    return [(f"Drishyam finds {d3['recall']:.2f} of the defects using {d3['pixels_frac']:.2f} of the image's pixels: "
            f"{hs['recall_ratio_vs_tiling']:.2f}× exhaustive tiling's recall (95 % CI {ci(hs['recall_ratio_vs_tiling_ci'])}) "
            f"at {hs['pixels_ratio_vs_tiling']:.2f}× its pixels"),
            (f"Drishyam is {hs['recall_ratio_vs_downscale']:.2f}× of it (95 % CI {ci(hs['recall_ratio_vs_downscale_ci'])}). "
            f"Chance scores {d3['chance_recall']:.2f}."),
            (f"({l3['recall']:.2f} against {lg['policies']['downscale'][0]['recall']:.2f}, {hl['recall_ratio_vs_downscale']:.2f}×, "
            f"95 % CI {ci(hl['recall_ratio_vs_downscale_ci'])})"),
            f"({lg['policies']['tiling'][0]['pixels_frac']:.2f} against {l3['pixels_frac']:.2f})",
            (f"Drishyam at 30 %: {c3['recall']:.2f}, chance {c3['chance_recall']:.2f}; IoU-only {c3['recall_iou_only']:.2f}, "
            f"chance {c3['chance_recall_iou_only']:.2f}"),
            (f"{hc['recall_ratio_vs_tiling']:.2f}× exhaustive tiling (95 % CI {ci(hc['recall_ratio_vs_tiling_ci'])}) and "
            f"{hc['recall_ratio_vs_downscale']:.2f}× one downscale ({ci(hc['recall_ratio_vs_downscale_ci'])})"),
            (f"({a3['synth200']['no-survey']['ratio_vs_full']:.2f}× the full agent on the 2048×1536 set, 95 % CI "
            f"{ci(a3['synth200']['no-survey']['ratio_vs_full_ci'])}; {a3['synth_large']['no-survey']['ratio_vs_full']:.2f}× at "
            f"4608×3456; {a3['codebrim']['no-survey']['ratio_vs_full']:.2f}× on CODEBRIM)"),
            (f"(pooled 95 % CI {pooled['synth_large']}) and on CODEBRIM ({pooled['codebrim']}) but matches it on the "
            f"2048×1536 set ({pooled['synth200']})"),
            f"(without them {q(a1, 'no-uncertainty')}, and {q(a1, 'no-seed')})",
            f"({q(a3['synth200'], 'no-verify')}) and lower uncapped there ({q(a1, 'no-verify')})"]


def fallback_quotes() -> list:
    """Section 9's account of the zero-candidate fallback's dev rule (results/dev/zero_candidate_fallback.json)."""
    z = load("results/dev/zero_candidate_fallback.json")
    a, ev = z["acted_on"], z["evidence"]["real"]
    n = sum(x["photos"] for x in a.values())
    assert z["verdict"]["adopted"] and n == 1 and a["synth_dev"]["photos"] == a["clean"]["photos"] == 0
    r = a["real"]
    return [f"that pass rests on {n} of {sum(x['of'] for x in a.values())} dev photos",
            (f"drafted {r['boxes_drafted']} crack boxes and found its {r['ground_truth_boxes_on_them']} labelled box, which "
             f"the run without it missed: recall ×{ev['vs_ref']} (CI {ev['vs_ref_ci'][0]:.2f}–{ev['vs_ref_ci'][1]}) over "
             f"CODEBRIM dev100, +{r['extra_modelled_ms_per_photo_acted_on']} modelled ms on that photo, no critical box")]


def test_report_prose_quotes_are_the_json():
    assert [w for w in report_quotes() if w not in flat(REPORT)] == []
    assert [w for w in backlog_quotes()[1:3] if w not in flat(README)] == []
    assert [w for w in backlog_quotes()[3:] + fallback_quotes() if w not in flat(REPORT)] == []
    section8 = REPORT.split("## 8. Evaluation")[1].split("\n## 9.")[0]
    assert [w for w in policy2_quotes() + policy2_cost_quotes() if w not in flat(section8)] == []
    assert [r for r in policy2_rows() + policy2_gate_rows() if r not in section8] == []
    assert [w for w in readme_quotes() if w not in flat(README)] == []


def test_real_photo_miss_is_the_trace():
    """The IMG_2332 failure case: the v1 run on real_01 proposed no candidate, took no look and drafted nothing."""
    recs = [json.loads(line) for line in (REPO / "public/samples/real_01.ndjson").read_text().splitlines()]
    ov = next(r for r in recs if r["type"] == "overview")
    done = next(r for r in recs if r["type"] == "done")
    assert ov["candidates"] == [] and done["cause"] == "exhausted" and done["summary"]["steps"] == 0
    assert done["summary"]["findings"] == 0 and load("video/live/real.json")["image"] == "IMG_2332"
    # its work order does not read as a clean wall: the whole photo is not assessed, and the order is incomplete
    wo, job = done["work_order"], recs[0]
    (na,) = wo["not_assessed"]
    assert wo["findings"] == [] and na["box"] == [0, 0, job["width"], job["height"]]
    assert f"lists the whole photo as `not_assessed` (action `{na['action']}`) and reads `{wo['status']}`, not `ready`" in flat(REPORT)
    assert f"{ov['pixels_frac']:.4f} of the pixels" in flat(REPORT) and "public/samples/real_01.jpg" in REPORT
    other = [json.loads(line) for line in (REPO / "public/samples/real_02.ndjson").read_text().splitlines()][-1]
    (f,) = other["work_order"]["findings"]  # IMG_2330: one look, one crack
    seen = load("video/live/variants/IMG_2330/real.json")["seen"]["paths"]
    lower = max(seen, key=lambda path: min(y for _, y in path))
    assert f["cls"] == "crack" and min(y for _, y in lower) <= f["box"][1] + f["box"][3]
    assert f"a {f['box'][2]} × {f['box'][3]} px box on part of the lower crack" in flat(REPORT)


def test_site_states_v1_policy2_as_the_json():
    """The site's FIG. 4.2 note on the live look policy quotes the same held-out read as the glance blocks."""
    html = (REPO / "public/index.html").read_text()
    note = html.split('id="policy2-note">')[1].split("</p>")[0]
    q, (_, cb, _, pr) = policy2_quotes(), p2()
    m, mv, crit = cb["matrix"]["v1-policy2@0.3"], cb["matrix"]["v1@0.3"], pr["codebrim@0.3: v1-policy2 / v1"]["critical"]
    gate = (f"{m['critical_per_image']:.2f} critical boxes per photo to a person instead of {mv['critical_per_image']:.2f} "
            f"({crit[0]:.2f}×, CI {ci(crit[1])}) and about {m['signoffs_per_image']:.2f} critical findings instead of "
            f"{mv['signoffs_per_image']:.2f}")
    assert [w for w in (gate, q[2], q[5]) if w not in note] == []


def test_docs_link_only_files_that_exist():
    for path, text in (("README.md", README), ("docs/report.md", REPORT)):
        base = (REPO / path).parent
        links = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text)
        missing = [x for x in links if "://" not in x and not (base / x).exists()]
        assert missing == [], path


def test_no_codebrim_image_is_published():
    """CODEBRIM's licence bars sharing its photos: no image file under a CODEBRIM path is tracked."""
    import subprocess

    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    assert [f for f in files if "codebrim" in f.lower() and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))] == []


if __name__ == "__main__":
    for k, f in ROWS.items():
        print(f"\n{k}", *f(), sep="\n")
    print("\nglance", *glance_rows(), sep="\n")
    print("", *classical_quotes(), *heldout_quotes(), *live_quotes(), *adoption_quotes(), *arm_quotes(), sep="\n")
