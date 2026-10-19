"""The narration of the submission video, written against the numbers build.py reads from the result files.

lines(D) returns one dict per spoken line: scene, id, say (what the voice reads) and cap (the burned-in caption).
Both are joined from the same parts: a part is plain text (said and printed alike) or a pair (spoken, printed),
so a caption is the spoken line with its numbers and code names printed ("zero point nine four" -> "0.94"),
never other words or other facts. Every number comes from D, never from here; a claim whose truth depends on
the data is checked by an assert, so a re-run that changes the result stops the build instead of shipping a
sentence that is no longer true. Exact figures and sources go on the boards, not in the captions.
"""

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

    # 3. problem and user
    assert num(imp["months"]) == 24, "p1 says every two years"
    add("problem", "p1", "America has about ", W(round(num(imp["bridges"]), -3)), " bridges, each due a routine inspection at least every two years.")
    hund = num(imp["photos"]) // 100   # 2,536 -> "twenty-five hundred": a figure costs the words it takes to say
    add("problem", "p2", "In an Oregon study, the largest drone inspection produced more than ", (f"{words(hund)} hundred", f"{hund * 100:,}"), " photos of one bridge.")
    add("problem", "p3", "Drishyam is for the engineer reviewing them: it zooms in where needed, and hands critical findings to a person.")

    # 4. the app working, live
    add("live", "l1", "The live site runs on AWS: OpenCV ", CV, ", on an arm Lambda.")
    assert sum(g["privacy"]) == 0, "l2 says there was nothing to blur in this photo"
    add("live", "l2", "Watch it inspect a bridge. A privacy pass runs first; nothing to blur here.")
    add("live", "l3", f"One survey look at the whole photo, at one {ORD[g['survey_div']]} size.")
    assert g["tau"] == 0.5 and g["entropy1"] > g["tau"]
    add("live", "l4", "Entropy ", Dc(g["entropy1"]), " is above ", ("one half", "0.5"), ". Unsure, so it zooms in.")
    add("live", "l5", "Then it moves to the best region, and double-checks at full resolution.")
    assert g["iou"] < g["iou_min"] == 0.5 and g["rebar_rule"].startswith("exposed_rebar: always critical")
    add("live", "l6", "At full resolution the masks disagree: unconfirmed. Exposed rebar always goes to a person.")
    assert g["missed"] == ["crack"] and len(g["unvisited"]) == 2
    add("live", "l7", "Then a second spall. It stops when the next look costs more than it is worth, and so it misses a crack in a region it never visited.")
    assert g["seconds"] < 1, "l8 says under a second"
    add("live", "l8", "The agent took under a second; the page replays it slowly.")
    add("live", "l9", "Only a person can approve. A name, Approve, and the server signs the work order.")
    add("live", "l11", "Change one character of that signature, and the server refuses it.")

    # 5. failure handling and human control
    add("break", "b0", "Now, break it.")
    add("break", "b1", f"Slow the segmenter down, and the governor switches to the {br['model']} model.")
    add("break", "b2", "The planner drill sends a shell command. The allow-list rejects it, and the rule planner decides.")
    add("break", "b3", "KILL halts the job before its next look; Resume replays the moves so far and carries on.")
    add("break", "b4", "No language model runs live. These drill planners are scripted to fail.")

    # 6. an LLM driving the OpenCV tools over MCP
    assert m["iou_a"] < 0.5 <= m["iou_b"], m
    add("mcp", "m1", "Over MCP, Claude drove the same OpenCV tools on the same photo.")
    add("mcp", "m2", "At call ", W(m["call_a"]), ", the full-resolution mask disagreed, and exposed rebar is critical: Claude handed it to a human.")
    add("mcp", "m3", "At call ", W(m["call_b"]), ", the masks agreed: a spall, confirmed. At call ", W(m["call_c"]), ", the budget ended the search.")
    add("mcp", "m4", "It still missed the large spall. The rules are in the prompt; the values were measured.")

    # 7. architecture
    add("arch", "a0", "Every step is perception, decision and action, with a human gate at the end.")
    add("arch", "a1", "CloudFront and a rate-limiting firewall front a signed Function URL and the arm Lambda.")
    add("arch", "a2", "DynamoDB holds the kill flags, Secrets Manager the signing key, and CloudWatch the alarms.")
    add("arch", "a3", "OpenCV does the looking: a pyramid, saliency, ", ("warp perspective", "warpPerspective"), " as a virtual pan-tilt-zoom camera, a blur gate, and face blurring.")
    assert h["segmenter"].startswith("classical"), "a4 says the live site uses the classical segmenter"
    add("arch", "a4", "Live, it uses the classical segmenter; OpenCV ", CV, "'s new DNN engine is on the optional ONNX path.")
    lo, hi = a["p95"]
    cap_ms = (int(hi) // 10 + 1) * 10   # 88.0 -> "under ninety"; the rail card prints the range
    add("arch", "a5", "On Lambda, a step took ", (f"under {words(cap_ms)} milliseconds", f"under {cap_ms} ms"), " at ", P95,
        ", over ", W(a["smoke_runs"]), " smoke runs. COOL was not run.")

    # 8. principal results
    assert not s["meets_target"] and s["cap"] == 0.3
    add("results", "r1", "On ", W(s["n"]), " synthetic photos, it finds ", P(s["ratio"]), " of exhaustive tiling's recall, at ", P(s["head"]["pixels_frac"]), " of the pixels: short of the ", P(s["target"]),
        " target, " + ("though not significantly." if s["ratio_ci"][1] >= s["target"] else "and so is its whole confidence interval."))
    assert s["ds_ci"][0] < 1.0 and s["ds_ratio"] < 1 and s["ds"]["recall"] > s["head"]["recall"], "r3 says the downscale is slightly ahead"
    add("results", "r3", "Against one downscaled pass it scores ", (f"{dec(s['ds_ratio'])} times", f"{s['ds_ratio']:.2f}×"), ": "
        + ("the downscale does as well, or slightly better." if s["ds_ci"][1] <= 1.0 else "slightly behind, within the noise."))
    assert lg["ds_ci"][0] > 1.0 and lg["l2"]["recall"] >= lg["head"]["recall"] and lg["l2"]["pixels_frac"] < lg["head"]["pixels_frac"]
    add("results", "r4", f"On larger photos it beats one downscale, which reads a {ORD[round(lg['head']['pixels_frac'] / lg['ds']['pixels_frac'])]} as many pixels; "
        f"{ORD[lg['l2_div']]}-size tiling does as well with fewer.")
    assert all(ab[d]["no-survey"] < 1 and ab[d]["vs"]["no-survey"] == "below the full agent" for d in ab) and ab["synth200"]["random_lo"] >= 0.95
    assert all(abs(ab[d][v] - 1) < 0.02 and ab[d]["vs"][v] == "no measurable difference" for d in ab for v in ("no-uncertainty", "governor-on")), "r5 says these change nothing measurable"
    assert all(ab[d]["vs"]["no-verify"] in ("no measurable difference", "above the full agent") for d in ab), "r5 says dropping verify costs no recall"
    # complete by construction: every other variant (no verify, no uncertainty, governor on) is asserted above to lose no recall
    add("results", "r5", "Ablations: only dropping the survey look loses recall; the others lose none, and a random gaze order matches it on smaller photos.")
    assert cb["plain"] and cb["chance"] < cb["range"][0], "r6 says every method finds about 1 in N, and chance is below that"
    c8 = round(1 / cb["chance"])
    add("results", "r6", "On ", W(cb["n"]), " real CODEBRIM photos, every method finds about ", (f"one defect in {words(cb['one_in'])}", f"1 defect in {cb['one_in']}"),
        "; its boxes at random, about ", (f"one in {words(c8)}", f"1 in {c8}"), ".")
    assert abs(cl["any"] - 2 / 3) < 0.01, "r7 says two times in three"
    add("results", "r7", "On ", W(cl["n"]), " synthetic photos with no damage, it still reports something ", ("two times in three", "2 times in 3"), ".")
    if D.get("model"):   # the trained segmenter: build.py has checked governor off, same cost table and same data
        md = D["model"]
        wins = [(f"L {words(int(x[1:]))}", x) for x in md["tiling_wins"]]
        tail = (["Every tiling level still finds more."] if len(wins) == 3 else ["Tiling at ", wins[0], " still finds more."] if len(wins) == 1
                else ["Tiling at ", wins[0], " and ", wins[1], " still finds more."] if wins else ["No tiling level finds more."])
        c_in = round(1 / md["chance"])
        leak = [] if md["near_dups"] == 0 else [" ", W(md["near_dups"]), " of these photos have a near-duplicate in its training set."]   # the board shows the count either way
        add("results", "r8", "Governor off, same cost table: a trained segmenter finds ", P(md["recall"]), " of the defects, the classical engine ",
            (words(round(100 * md["classical"])), f"{round(100 * md['classical'])} %"), "; at random, one in ", (words(c_in), str(c_in)), ". ", *tail, *leak)
    else:
        add("results", "r8", "Results for a trained segmenter are pending.")

    # 8b. a real photo of our own: only when a recorded job on one of the team's wall photos exists (build.py real_run)
    if D.get("real"):
        R = D["real"]; n = R["n_looks"]; seen = R["seen"]
        assert n == len(R["looks"]) and R["n_found"] == len(R["found"]) and (n or not R["found"]), R
        missed = bool(seen) and not R["hit"]
        eng = "the trained segmenter" if R["kind"] == "onnx" else "the classical engine"
        parts = ["Our own photo of a painted wall, no ground truth: ", eng]
        if n == 0:
            assert R["candidates"] == 0, "q1 says it saw nothing worth a closer look"
            parts += [" read ", P(R["pixels_frac"]), " of it, saw nothing worth a look, and ", f"missed {seen['say']}." if missed else "reported nothing."]
        elif not R["found"]:
            parts += [" took ", W(n), " looks" if n > 1 else " look", " at ", P(R["pixels_frac"]), " of the pixels, reported nothing", f", and missed {seen['say']}." if missed else "."]
        else:
            cls = {}
            for x in R["found"]:
                cls[x["cls"]] = cls.get(x["cls"], 0) + 1
            from build import SAY_CLS
            said = [SAY_CLS[c][0] if k == 1 or SAY_CLS[c][0] == SAY_CLS[c][1] else f"{words(k)} {SAY_CLS[c][1]}" for c, k in cls.items()]
            listed = said[0] if len(said) == 1 else ", ".join(said[:-1]) + " and " + said[-1]
            k = len(R["wrong"]) if seen else 0
            # with the team's reading, the verdict says what it found (the card lists the classes); without it, the classes
            verdict = ([f" and missed {seen['say']}" if missed else f" and found {seen['say']}"] + ([", with ", W(k), " false alarm" if k == 1 else " false alarms"] if k else [])
                       if seen else [" and reported ", listed])
            parts += [" took ", W(n), " looks" if n > 1 else " look", " at ", P(R["pixels_frac"]), " of the pixels", *verdict, "."]
        add("real", "q1", *parts)

    # 9. failure cases
    fs = D["fails_said"]
    assert fs["glare_listed"] and fs["formwork_kept_coarse"], fs
    add("failures", "f1", "Failures, in the open: graffiti read as spalling, a formwork line as a crack, glare left for a re-shoot, and blur.")
    assert len(D["fails"]) == 4 and "built" in next(x for x in D["fails"] if x["cat"].startswith("Formwork")), D["fails"]
    add("failures", "f2", (words(fs["not_built"]).capitalize(), str(fs["not_built"])), " of these rules are not built yet; the straight-line test is, but let this one through at a coarse look.")
    rf = D["real_fail"]
    add("failures", "f3", f"On a real photo, {rf['gt']} was reported as {rf['got']}.")

    # 10. limits
    add("limits", "x1", "Pixels, not millimetres. AI-assisted, not an inspection. Only a person can approve.")

    # 11. reproduce
    add("repro", "y1", ("Make test, make eval", "make test, make eval"), ", and one script to deploy, test and destroy the AWS stack.")

    # 12. end
    add("end", "e1", "Drishyam. It looks where it matters, and asks a human first.")
    return L
