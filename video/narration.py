"""The narration of the submission video, written against the numbers build.py reads from the result files.

lines(D) returns one dict per spoken line: scene, id, say (what the voice reads) and cap (the burned-in caption).
Both are joined from the same parts: a part is plain text (said and printed alike) or a pair (spoken, printed),
so a caption is the spoken line with its numbers and code names printed ("zero point nine four" -> "0.94"),
never other words or other facts. Every number comes from D, never from here; a claim whose truth depends on
the data is checked by an assert, so a re-run that changes the result stops the build instead of shipping a
sentence that is no longer true. Exact figures and sources go on the boards, not in the captions.
"""

import math

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def words(n):
    n = int(n)
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    if n < 1000:
        return ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + words(n % 100))
    if n < 1_000_000:
        return words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + words(n % 1000))
    return str(n)


def num(s):
    return int(str(s).replace(",", ""))


def pct(x):
    return f"{round(100 * x)}"


def dec(x, d=2):
    """0.94 -> 'zero point nine four' (Kokoro reads '0.94' unevenly)."""
    s = f"{x:.{d}f}"
    whole, frac = s.split(".")
    return f"{words(int(whole))} point {' '.join(ONES[int(c)] for c in frac)}"


ORD = {2: "half", 3: "third", 4: "quarter", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth"}


def lines(D):
    L = []

    def add(scene, id, *parts):
        L.append({"scene": scene, "id": id, "say": "".join(p if isinstance(p, str) else p[0] for p in parts),
                  "cap": "".join(p if isinstance(p, str) else p[1] for p in parts)})

    W = lambda n: (words(n), f"{int(n):,}")
    P = lambda x: (f"{words(round(100 * x))} percent", f"{round(100 * x)} %")
    Dc = lambda x, d=2: (dec(x, d), f"{x:.{d}f}")
    P95 = ("p ninety-five", "p95")

    g, br, k, imp, h, m, a = D["guided"], D["break"], D["kill"], D["impact"], D["health"], D["mcp"], D["aws"]
    s, lg, cb, cl, ab = D["syn"], D["large"], D["codebrim"], D["clean"], D["abl"]
    assert h["opencv"].startswith("5.") and h["arch"] == "aarch64" and h["host"] == "aws-lambda", h
    CV = ("five", "5")   # OpenCV's major version, from /api/health

    # 1. cold open, over the live guided run
    assert round(1 / g["pixels_frac"]) == 4, "o1 says a quarter"
    add("open", "o1", "It read a quarter of this photo's pixels, and found ", W(g["critical"]), " critical findings.")
    add("open", "o2", "It will not sign them off. A person must.")

    # 2. team
    add("team", "t1", "Drishyam, an Agentic Vision entry to the OpenCV AI Competition, powered by AWS.")
    add("team", "t2", "Built by Vaibhav Jangid, lead engineer, and Bhavya Dhoot, design and presentation.")
    if D.get("team"):   # the portraits: the second photo is held 1 s before its looks (film.js HOLD), so t2 waits as long
        L[-1]["pause"] = 0.95

    # 3. problem and user
    assert num(imp["months"]) == 24, "p1 says every two years"
    add("problem", "p1", "America has about ", W(round(num(imp["bridges"]), -3)), " bridges, inspected at least every two years.")
    hund = num(imp["photos"]) // 100   # 2,536 -> "twenty-five hundred": a figure costs the words it takes to say
    add("problem", "p2", "One drone inspection took more than ", (f"{words(hund)} hundred", f"{hund * 100:,}"), " photos of one bridge.")
    add("problem", "p3", "Drishyam is for the engineer reviewing them: it zooms in only where needed.")

    # 4. the app working, live (recorded 2026-10-18, when the site ran the classical engine: a4 says so)
    add("live", "l1", "It runs live on AWS: OpenCV ", CV, " on an arm Lambda.")
    assert sum(g["privacy"]) == 0, "l2 says there was nothing to blur in this photo"
    add("live", "l2", "Watch it inspect a bridge; a privacy pass finds nothing to blur.")
    add("live", "l3", f"One survey look at the whole photo, at one {ORD[g['survey_div']]} size.")
    assert g["tau"] == 0.5 and g["entropy1"] > g["tau"]
    add("live", "l4", "Entropy ", Dc(g["entropy1"]), ", above ", ("one half", "0.5"), ": unsure, so it zooms in.")
    add("live", "l5", "It moves to the best region and double-checks at full resolution.")
    assert g["iou"] < g["iou_min"] == 0.5 and g["rebar_rule"].startswith("exposed_rebar: always critical")
    add("live", "l6", "The masks disagree: unconfirmed. Exposed rebar always goes to a person.")
    assert g["missed"] == ["crack"] and len(g["unvisited"]) == 2
    add("live", "l7", "A second spall; then it stops, the next look not worth its cost, and misses a crack it never visited.")
    add("live", "l9", "Only a person can approve; then the server signs the work order.")
    L[-1]["pause"] = 0.5   # the footage cuts from the end of the run to the sign-off here (build.py footage_warp)
    add("live", "l11", "Change one character of the signature: the server refuses it.")

    # 5. failure handling and human control
    add("break", "b0", "Now, break it.")
    add("break", "b1", f"Slow the segmenter, and the governor switches to the {br['model']} model.")
    add("break", "b2", "The planner drill sends a shell command; the allow-list rejects it.")
    add("break", "b3", "KILL halts it before the next look; Resume replays the moves and carries on.")
    add("break", "b4", "No language model runs live; these planners are scripted to fail.")

    # 6. an LLM driving the OpenCV tools over MCP (the rail note: the rules were in its prompt, the values measured)
    assert m["iou_a"] < 0.5 <= m["iou_b"], m
    add("mcp", "m1", "Over MCP, Claude drove the same tools on the same photo.")
    add("mcp", "m2", "At call ", W(m["call_a"]), " the rebar masks disagreed, so Claude handed it to a human.")
    add("mcp", "m3", "At call ", W(m["call_b"]), " they agreed on a spall; at call ", W(m["call_c"]), ", the budget ended the search.")
    add("mcp", "m4", "It still missed the large spall.")

    # 7. architecture
    add("arch", "a0", "Each step is perception, decision and action, with a human gate at the end.")
    add("arch", "a1", "CloudFront and a rate-limiting firewall front the arm Lambda; DynamoDB keeps kill flags, Secrets Manager the key.")
    add("arch", "a3", "OpenCV does the looking: a pyramid, saliency, ", ("warp perspective", "warpPerspective"), " as a pan-tilt-zoom camera, a blur gate, face blurring.")
    md = D.get("model")
    assert h["segmenter"].startswith("classical"), "a4 says the recording ran the classical segmenter"
    assert md and a["now"]["segmenter"] == D["model_segmenter"] and a["now"]["segmenter"].startswith("onnx:"), "a4 says the site now runs the trained segmenter"
    add("arch", "a4", "Recorded on the classical segmenter; the site now runs the trained one, on OpenCV ", CV, "'s new DNN engine.")
    lo, hi = a["p95"]; cap_ms = (int(hi) // 10 + 1) * 10   # 88.0 -> "under ninety"; the rail card prints both ranges
    cap_s = math.ceil(a["p95_v1"][1] / 100) / 10   # 1067.3 ms -> "under 1.1 seconds"
    assert a["p95_v1"][0] > hi, "a5 says a step now takes longer"
    add("arch", "a5", "On Lambda a step took ", (f"under {words(cap_ms)} milliseconds", f"under {cap_ms} ms"), " at ", P95, " then, and ",
        (f"under {dec(cap_s, 1)} seconds", f"under {cap_s:.1f} s"), " now.")   # COOL not run: on the rail card

    # 8. principal results: the agent's policy, on the classical engine
    assert not s["meets_target"] and s["cap"] == 0.3
    add("results", "r1", "On ", W(s["n"]), " synthetic photos it finds ", P(s["ratio"]), " of full tiling's recall at ", P(s["head"]["pixels_frac"]),
        " of the pixels, short of its ", P(s["target"]), " target.")
    assert s["ds_ci"][0] < 1.0 < s["ds_ci"][1] and s["ds"]["recall"] > s["head"]["recall"], "r3 says the downscale does about as well"
    add("results", "r3", "One downscaled pass does about as well.")
    # the 4608 px photos (zooming beats one downscale there, and L2 tiling matches it) are on the board, not in the voice
    assert all(ab[d]["no-survey"] < 1 and ab[d]["vs"]["no-survey"] == "below the full agent" for d in ab) and ab["synth200"]["random_lo"] >= 0.95
    assert all(abs(ab[d][v] - 1) < 0.02 and ab[d]["vs"][v] == "no measurable difference" for d in ab for v in ("no-uncertainty", "governor-on")), "r5 says these change nothing measurable"
    assert all(ab[d]["vs"]["no-verify"] in ("no measurable difference", "above the full agent") for d in ab), "r5 says dropping verify costs no recall"
    # complete by construction: every other variant (no verify, no uncertainty, governor on) is asserted above to lose no recall
    add("results", "r5", "Ablations: only dropping the survey look loses recall, and a random gaze order matches it on smaller photos.")
    assert cb["plain"] and cb["chance"] < cb["range"][0], "r6 says every method finds about 1 in N, and chance is below that"
    c8 = round(1 / cb["chance"])
    add("results", "r6", "On ", W(cb["n"]), " real CODEBRIM photos, every classical method finds about ", (f"one defect in {words(cb['one_in'])}", f"1 defect in {cb['one_in']}"),
        "; chance, ", (f"one in {words(c8)}", f"1 in {c8}"), ".")
    # the trained segmenter, held out: build.py has checked the same photos, the governor off, and cost tables timed on one machine
    r, (rlo, _) = md["ratio"]; cr, (crlo, _) = md["crit_ratio"]
    assert rlo > 1 and abs(r - 2.5) < 0.1, "r8 says two and a half times as many"
    assert md["chance"] > md["classical"] and crlo > 1 and abs(cr - 5.5) < 0.1, "r8 says its boxes at random find more than the classical engine does, and five and a half times the critical boxes"
    assert abs(md["recall"] - md["classical"] * r) < 0.02, (md["recall"], md["classical"], r)
    leak = [] if md["near_dups"] == 0 else [" ", W(md["near_dups"]), " of these photos have a near-duplicate in its training set."]   # the board shows the count either way
    add("results", "r8", "Held out, the trained segmenter finds ", P(md["recall"]), ", two and a half times as many; but its boxes find ", P(md["chance"]),
        " at random, and a person gets five and a half times the critical boxes.", *leak)
    assert abs(cl["any"] - 2 / 3) < 0.01 and md["clean_any"] == [0.0, cl["any"]], "r7 says two times in three, and never for the trained segmenter"
    add("results", "r7", "On ", W(cl["n"]), " undamaged synthetic photos, the classical engine flags something ", ("two times in three", "2 times in 3"),
        "; the trained one, never.")

    # 8b. two real photos of our own, each a recorded run of the live engine (build.py real_run): each line built from its trace
    PR = lambda x: (f"{dec(100 * x, 1)} percent", f"{100 * x:.1f} %") if x < 0.1 else P(x)   # as the PIXELS READ card prints it
    for i, R in enumerate(D.get("real") or []):
        n, seen = R["n_looks"], R["seen"]
        assert n == len(R["looks"]) and R["n_found"] == len(R["found"]) and (n or not R["found"]), R
        assert R["kind"] == "onnx" and R["engine"] == D["model_segmenter"], "q lines say the trained segmenter"
        lead = "Our own wall, no ground truth: " if i == 0 else "On a second wall, "
        if n == 0:   # the overview (saliency at a coarse level, no segmenter) proposed nothing, so no look ran the segmenter
            assert R["candidates"] == 0 and not R["found"] and seen and not any(R["paths_hit"]), "q says no region, nothing flagged, the crack missed"
            parts = [lead, "no region was proposed, so the trained segmenter never ran, and missed ", seen["say"], "."]
            beats = {"eng": "wall", "read": "proposed", "rep": "never ran", "end": "missed"}
        else:
            ph = R["paths_hit"]
            assert seen and len(ph) == 2 and sum(ph) == 1 and len(R["found"]) == 1 and not R["wrong"] and R["coverage"] < 0.5, \
                "q says one finding, on one of the two lines the team traced, and most of the wall never looked at closely"
            which = "lower" if R["path_y"][ph.index(True)] > R["path_y"][ph.index(False)] else "upper"
            parts = [lead, "one close look flagged a crack on the ", which, " of two we traced, and never looked closely at most of the wall."]
            beats = {"eng": "wall", "read": "close look", "rep": "flagged", "end": which}
        add("real", f"q{i + 1}", *parts)
        L[-1]["beats"] = beats
    # 9. failure cases
    fs = D["fails_said"]
    assert fs["glare_listed"] and fs["formwork_kept_coarse"], fs
    add("failures", "f1", "Failures in the open: graffiti read as spalling, a formwork line as a crack, glare, and blur.")
    assert len(D["fails"]) == 4 and "built" in next(x for x in D["fails"] if x["cat"].startswith("Formwork")), D["fails"]
    add("failures", "f2", (words(fs["not_built"]).capitalize(), str(fs["not_built"])), " of these four rules are not built yet.")
    # the formwork plate's built-but-passed rule, and the CODEBRIM case in words (D["real_fail"]), are on the board

    # 10. limits
    add("limits", "x1", "Pixels, not millimetres. AI-assisted, not an inspection. Only a person can approve.")

    # 11. reproduce
    add("repro", "y1", ("Make test, make eval", "make test, make eval"), ", and one script for the AWS stack.")

    # 12. end
    add("end", "e1", "Drishyam. It looks where it matters, and asks a human first.")
    return L
