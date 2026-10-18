"""The narration of the submission video, written against the numbers build.py reads from the result files.

lines(D) returns one dict per spoken line: scene, id, say (what the voice reads), cap (the burned-in caption,
digits and symbols as printed in the source; defaults to say). Every number comes from D, never from here;
a claim whose truth depends on the data is checked by an assert, so a re-run that changes the result stops
the build instead of shipping a sentence that is no longer true. Spoken numbers are rounded the way a person
says them ("about six hundred twenty-five thousand"); the caption under them prints the exact figure.
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


def lines(D):
    L = []

    def add(scene, id, say, cap=None):
        L.append({"scene": scene, "id": id, "say": say, "cap": cap or say})

    g, br, k, imp, h, m, a = D["guided"], D["break"], D["kill"], D["impact"], D["health"], D["mcp"], D["aws"]
    s, lg, cb, cl, ab = D["syn"], D["large"], D["codebrim"], D["clean"], D["abl"]

    # 1. cold open, over the live guided run
    add("open", "o1", f"It read {pct(g['pixels_frac'])} percent of this photo's pixels, and found {words(g['critical'])} critical findings.",
        f"It read {g['pixels_frac'] * 100:.1f} % of this photo's pixels and found {g['critical']} critical findings.")
    add("open", "o2", "It will not sign them off. A person must.")

    # 2. team
    add("team", "t1", "Drishyam is an Agentic Vision entry to the OpenCV AI Competition, powered by AWS.",
        "Drishyam: an Agentic Vision entry to the OpenCV AI Competition 2026, powered by AWS.")
    add("team", "t2", "Built by Vaibhav Jangid, lead engineer, and Bhavya Dhoot, design and presentation.")

    # 3. problem and user
    bridges = num(imp["bridges"])
    add("problem", "p1", f"America has about {words(round(bridges, -3))} bridges. As a rule, each gets a routine inspection at intervals of up to {words(num(imp['months']))} months.",
        f"The US has {imp['bridges']} bridges ({imp['poor']} in poor condition); the rule: a routine inspection at intervals of up to {imp['months']} months.")
    photos = num(imp["photos"])
    add("problem", "p2", f"One drone inspection of a single bridge produced more than {words(photos // 100 * 100)} photos. Someone has to look at every one.",
        f"One drone inspection of one bridge produced {imp['photos']} photos. Someone has to look at every one.")
    add("problem", "p3", "Drishyam is for the engineer reviewing them. It zooms in only where it needs to, and hands every critical finding to a person.")

    # 4. the app working, live
    assert h["opencv"] == "5.0.0" and h["arch"] == "aarch64" and h["host"] == "aws-lambda", h
    add("live", "l1", "The live site runs on AWS: OpenCV five, on an arm Lambda.",
        f"Live on AWS: opencv {h['opencv']} · arch {h['arch']} · host {h['host']}")
    add("live", "l2", "Watch it inspect a bridge. Faces are blurred first.")
    add("live", "l3", "One survey look at the whole photo, at one eighth size.")
    add("live", "l4", f"Entropy {dec(g['entropy1'])} is above one half. Unsure, so it zooms in.",
        f"Entropy {g['entropy1']:.2f} > 0.5: unsure, so it zooms in one level.")
    add("live", "l5", "Then it moves to the best region, and double-checks at full resolution.")
    assert g["iou"] < 0.5
    add("live", "l6", f"The two looks barely overlap, {dec(g['iou'], 3)}. So the exposed rebar goes to a person.",
        f"The two looks barely overlap (IoU {g['iou']:.3f} < 0.5), so the exposed rebar goes to a person.")
    add("live", "l7", "Then a second spall, and it stops when the next look would cost more than it is worth.")
    add("live", "l8", f"The agent took {dec(g['seconds'], 1) if g['seconds'] < 1 else g['seconds']} seconds. The page replays it slowly.",
        f"The agent took {g['seconds']} s; the page replays it slowly.")
    add("live", "l9", "Only a person can approve. A name, Approve, and the server signs the work order.")
    t = D["tamper"]
    add("live", "l11", "Change one character of that signature, and the server refuses it.",
        f"One character of the signature changed: HTTP {t['status']}, “{t['error']}”.")

    # 5. failure handling and human control
    add("break", "b0", "Now, break it.")
    add("break", "b1", f"A slowed segmenter pushes its p ninety-five to {words(round(br['p95']))} milliseconds, so the governor switches to the fast model.",
        f"Slowed segmenter: p95 {round(br['p95'])} ms, so the governor switches to the {br['model']} model.")
    add("break", "b2", "The planner drill sends a shell command. The allow-list rejects it, and the rule planner decides.",
        f"The planner drill sends “{br['planner_bad']}”: rejected by the allow-list; the rule planner decides.")
    add("break", "b3", "KILL halts the job before its next look. Resume replays the moves so far, and carries on.",
        f"KILL: halted, cause {k['cause']}. Resume: {D['resume']['replayed']} moves replayed, then it carries on.")
    add("break", "b4", "No language model runs live. These drill planners are scripted to fail.")

    # 6. an LLM driving the OpenCV tools over MCP
    assert m["iou_a"] < 0.5 <= m["iou_b"], m
    add("mcp", "m1", "Over MCP, Claude drove the same OpenCV tools on the same photo, one call at a time.",
        "Recorded MCP session: Claude drives the same OpenCV tools on bridge_03, one call at a time.")
    add("mcp", "m2", f"At call {words(m['call_a'])}, two zoom levels overlapped by {dec(m['iou_a'])}. So Claude handed the rebar to a human.",
        f"Call {m['call_a']}: IoU {m['iou_a']:.2f} between L1 and L0, so the rebar goes to a human.")
    add("mcp", "m3", f"At call {words(m['call_b'])}, {dec(m['iou_b'])} confirmed a spall. At call {words(m['call_c'])}, the budget ended the search.",
        f"Call {m['call_b']}: IoU {m['iou_b']:.2f} confirms a spall. Call {m['call_c']}: the measured budget ends the search.")
    add("mcp", "m4", "It still missed the large spall. The rules are in the prompt; the values were measured.")

    # 7. architecture
    add("arch", "a0", "Every step is perception, decision and action, with a human gate at the end.")
    add("arch", "a1", "CloudFront and a rate-limiting firewall front a signed Function URL, and an arm Lambda with stock OpenCV five.",
        f"CloudFront + WAF ({a['waf_n']} job starts per IP per {a['waf_s']} s) → Function URL (AWS_IAM) → Lambda arm64, stock OpenCV 5.0.0")
    add("arch", "a2", "DynamoDB holds the kill flags, Secrets Manager the signing key, and CloudWatch the alarms.")
    add("arch", "a3", "OpenCV does the looking: a pyramid, saliency, warp perspective as a virtual pan-tilt-zoom camera, a blur gate, and face blurring.",
        "OpenCV: pyrDown pyramid · saliency (Sobel, black-hat, CLAHE) · warpPerspective virtual PTZ · Laplacian gate · YuNet + NanoDet blur")
    add("arch", "a4", "The live demo uses the classical segmenter. OpenCV five's new DNN engine is on the optional ONNX path.",
        "Live: the classical segmenter. The one OpenCV-5-only call, cv2.dnn.ENGINE_NEW, is on the optional ONNX path.")
    add("arch", "a5", f"On Lambda, a step took {words(round(a['p95']))} milliseconds at p ninety-five. COOL was not run.",
        f"Lambda step p95: {a['p95']} ms (last smoke test). COOL: not run, the Marketplace AMI could not be subscribed.")

    # 8. principal results
    assert not s["meets_target"]
    add("results", "r1", f"On {words(s['n'])} synthetic photos, under a thirty percent pixel cap, it finds {words(round(100 * s['ratio']))} percent of exhaustive tiling's recall, at {words(round(100 * s['head']['pixels_frac']))} percent of the pixels. The {words(round(100 * s['target']))} percent target was missed.",
        f"{s['n']} synthetic photos, 30 % cap: {pct(s['ratio'])} % of exhaustive tiling's recall at {pct(s['head']['pixels_frac'])} % of the pixels. Target {pct(s['target'])} %: missed.")
    assert s["ds_ci"][0] <= 1.0 <= s["ds_ci"][1], "the downscale sentence says 'about as well'"
    add("results", "r3", f"And one downscaled pass does about as well: {dec(s['ds_ratio'])} times.",
        f"One downscale does about as well: {s['ds_ratio']:.2f}× (95 % CI {s['ds_ci'][0]:.2f}–{s['ds_ci'][1]:.2f}).")
    assert lg["ds_ci"][0] > 1.0 and lg["l2"]["recall"] >= lg["head"]["recall"] and lg["l2"]["pixels_frac"] < lg["head"]["pixels_frac"]
    add("results", "r4", "On larger photos it does beat one downscale, but quarter-size tiling matches it with fewer pixels.",
        f"4608 px: {lg['ds_ratio']:.2f}× one downscale (CI {lg['ds_ci'][0]:.2f}–{lg['ds_ci'][1]:.2f}); L2 tiling {lg['l2']['recall']:.2f} at {lg['l2']['pixels_frac']:.2f} of the pixels, ours {lg['head']['recall']:.2f} at {lg['head']['pixels_frac']:.2f}.")
    assert all(ab[d]["no-survey"] < 1 for d in ab) and ab["synth200"]["random_lo"] >= 0.95
    add("results", "r5", "In the ablations, dropping the survey look hurts on every set. A random gaze order does as well on the smaller photos.")
    assert cb["plain"]
    add("results", "r6", f"On {words(cb['n'])} real CODEBRIM photos, every method finds about one defect in {words(cb['one_in'])}. The classical segmenter is the bottleneck.",
        f"CODEBRIM, {cb['n']} real photos: every method finds about 1 in {cb['one_in']} defects; its boxes at random score {cb['chance']:.2f}.")
    add("results", "r7", f"And on photos with no damage, it still reports something {words(round(100 * cl['any']))} percent of the time.",
        f"{cl['n']} photos with no damage: something reported on {pct(cl['any'])} % of them ({cl['boxes']:.2f} boxes per image).")
    if D.get("model"):
        md = D["model"]
        add("results", "r8", f"With the trained segmenter, on the same real photos, recall is {dec(md['recall'])}.",
            f"Trained segmenter on CODEBRIM: recall {md['recall']:.2f} at {pct(md['pixels_frac'])} % of the pixels.")
    else:
        add("results", "r8", "Results for a trained segmenter are pending.", "Trained segmenter: results pending.")

    # 9. failure cases
    add("failures", "f1", "The failures stay in the open: graffiti read as spalling, a formwork line as a crack, glare, and blur.")
    add("failures", "f2", "Each names the rule that would catch it. None of them is built yet.")
    rf = D["real_fail"]
    add("failures", "f3", f"On a real photo, {rf['gt']} was reported as {rf['got']}.",
        f"CODEBRIM {rf['id']} (text only): ground truth {rf['gt']}, reported {rf['got']}.")

    # 10. limits
    add("limits", "x1", "Pixels, not millimetres. AI-assisted, not an inspection. Only a person signs.")

    # 11. reproduce
    add("repro", "y1", "Make test, make eval, and one script that deploys, tests and destroys the AWS stack.")

    # 12. end
    add("end", "e1", "Drishyam. It looks where it matters, and asks a human first.")
    return L
