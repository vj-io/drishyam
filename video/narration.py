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
    P95 = ("p ninety-five", "p95")
    RB = ("reebar", "rebar")   # Kokoro reads "rebar" as ri-BAR (its phonemes); respelt for the voice only

    g, br, k, imp, h, m, a = D["guided"], D["break"], D["kill"], D["impact"], D["health"], D["mcp"], D["aws"]
    s, lg, cb, cl, ab = D["syn"], D["large"], D["codebrim"], D["clean"], D["abl"]
    assert h["opencv"].startswith("5.") and h["arch"] == "aarch64" and h["host"] == "aws-lambda", h
    CV = ("five", "5")   # OpenCV's major version, from /api/health

    # 1. cold open, over the live guided run (recorded 2026-10-22 on the live site's trained v1: o1 says which, l0 says when)
    assert round(1 / g["pixels_frac"]) == 4, "o1 says a quarter"
    assert h["segmenter"] == D["model_segmenter"] and h["segmenter"].startswith("onnx:"), "o1 and l0 say the footage ran the trained engine"
    assert g["critical"] == 1, "o1 says one critical finding, o2 'it'"
    add("open", "o1", "Its trained engine read a quarter of the pixels and flagged ", W(g["critical"]), " critical finding.")
    add("open", "o2", "It will not sign it off. A person must.")

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

    # 4. the app working, live (recorded 2026-10-22 on the site as it runs now: the trained v1 with v1-policy2, l0 says so before it plays)
    md = D.get("model")
    assert D["lit"]["recorded"] == "2026-10-22", "l0 says October twenty-second"
    assert md and md["name"] == "v1" and a["now"]["segmenter"] == D["model_segmenter"] and a["now"]["segmenter"].startswith("onnx:"), "l0 and a4 say the site now runs the trained v1"
    assert h["segmenter"] == a["now"]["segmenter"], "l0: the footage ran the engine the live site runs (its last smoke run's /api/health)"
    assert "v1-policy2 look policy since 2026-10-22" in D["lit"]["policy"], "l0 says its look policy is new"
    add("live", "l0", "Recorded ", ("October twenty-second", "22 October"), " on the live site: the trained v1 and its new look policy.")
    add("live", "l1", "Live on AWS: OpenCV ", CV, " on an arm Lambda.")
    assert sum(g["privacy"]) == 0, "l2 says there was nothing to blur in this photo"
    add("live", "l2", "Watch it inspect a bridge; a privacy pass finds nothing to blur.")
    assert g["survey_found"] == ["corrosion", "spalling"], "l3 says it finds spalling and corrosion"
    add("live", "l3", f"One survey look at the whole photo, at one {ORD[g['survey_div']]} size, finds spalling and corrosion.")
    # v1-policy2: the rebar's confidence is under the floor, so it is held for one confirm look (build.py data() asserts the trace)
    (ccls, cstep, _), (hcls, hconf, floor, hlev) = g["crit"], g["held"]
    assert ccls == "spalling" and hcls == "exposed_rebar" and hconf < floor and hlev == g["survey_level"] - 1 and cstep == 2, (g["crit"], g["held"])
    add("live", "l4", "One level finer, the spalling is big enough to be critical, but the ", RB, " scores under the confidence floor.")
    assert g["confirm_level"] == hlev - 1
    add("live", "l5", "It takes one confirm look, centred on the ", RB, ", and does not confirm it.")
    assert "not a critical finding; not confirmed" in g["eye"] and g["gt"][1] == ["exposed_rebar"], "l6 says the rebar goes to a person, not as a critical finding"
    add("live", "l6", "So the ", RB, " goes to a person to check by eye, not as a critical finding.")
    assert g["after"] == [3, ["crack"]] and g["stop"][0] == "exhausted" and g["stop"][2] == [] and "crack" in g["gt"][0], g["after"]
    add("live", "l7", "Three more looks find a crack; with every region looked at, it stops.")
    assert g["order"] == ["incomplete", g["critical"]], "l9 says the order stays incomplete after the one approval"
    add("live", "l9", "Only a person can approve the spalling; the order stays incomplete until someone checks the ", RB, ".")
    L[-1]["pause"] = 0.5   # the footage cuts from the end of the run to the sign-off here (build.py footage_warp)
    add("live", "l11", "Change one character of the order's signature: the server refuses it.")

    # 5. failure handling and human control
    add("break", "b0", "Now, break it.")
    add("break", "b1", f"Slow the segmenter, and the governor switches to the {br['model']} model.")
    add("break", "b2", "The planner drill sends a shell command; the allow-list rejects it.")
    assert D["kill"]["cause"] == "kill-switch" and D["resume"]["replayed"] == D["kill"]["steps"] < D["resume"]["looks"], (D["kill"], D["resume"])
    add("break", "b3", "KILL halts it before the next look; Resume replays the moves and carries on.")   # where Resume stopped is on the rail card
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
    add("arch", "a4", "The site now runs v1 on OpenCV ", CV, "'s new DNN engine.")
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
    # real photos: the trained v1, held out, is the headline; the classical engine (every classical method about 1 in N,
    # above its own chance, on the board's comparison row) is what it is compared with. build.py has checked the same
    # photos, the governor off, and cost tables timed on one machine.
    assert cb["plain"] and cb["chance"] < cb["range"][0] and cb["n"] == md["n"], "the board says every classical method finds about 1 in N, above chance, on these photos"
    r, (rlo, _) = md["ratio"]; cr, (crlo, _) = md["crit_ratio"]
    assert rlo > 1 and abs(r - 2.5) < 0.1 and round(1 / md["classical"]) == cb["one_in"], "r6 says two and a half times the classical engine's one in N"
    assert abs(md["recall"] - md["classical"] * r) < 0.02, (md["recall"], md["classical"], r)
    add("results", "r6", "On ", W(cb["n"]), " real CODEBRIM photos, held out, the trained v1 finds ", P(md["recall"]), ": two and a half times the classical engine's ",
        (f"one in {words(cb['one_in'])}", f"1 in {cb['one_in']}"), ".")
    assert md["chance"] > md["classical"] and crlo > 1 and abs(cr - 5.5) < 0.1, "r8 says its boxes at random find a lot, and five and a half times the critical boxes"
    leak = [] if md["near_dups"] == 0 else [" ", W(md["near_dups"]), " of these photos have a near-duplicate in its training set."]   # the board shows the count either way
    add("results", "r8", "But its boxes find ", P(md["chance"]), " at random, and a person gets five and a half times the critical boxes.", *leak)
    assert abs(cl["any"] - 2 / 3) < 0.01 and md["clean_any"] == [0.0, cl["any"]], "r7 says two in three, and none for v1"
    add("results", "r7", "On ", W(cl["n"]), " undamaged synthetic photos, the classical engine flags ", ("two in three", "2 in 3"), "; v1, none.")

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
    # failures.json: only the glare gate caught its case; graffiti and blur have no rule, and the formwork case needs the
    # full-resolution check that is "Not implemented" (its straight-line test is built and kept the line)
    assert fs["caught"] == ["Glare caught by the quality gate"] and fs["not_fixed"] == 3, fs
    add("failures", "f2", "Only the glare gate caught its case; ", words(fs["not_fixed"]), " fixes are not built yet.")
    # the formwork plate's built-but-passed rule, and the CODEBRIM case in words (D["real_fail"]), are on the board

    # 10. limits
    add("limits", "x1", "Pixels, not millimetres. AI-assisted, not an inspection. Only a person can approve.")

    # 11. reproduce
    add("repro", "y1", ("Make test, make eval", "make test, make eval"), ", and one script for the AWS stack.")

    # 12. end
    add("end", "e1", "Drishyam. It looks where it matters, and asks a human first.")
    return L
