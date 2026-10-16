import json

import pytest

from drishyam import governor as G
from drishyam import planner as P

DIGEST = {"step": 5, "max_level": 3, "looks_left": 4, "pixels_left_frac": 0.5, "usd_left": 0.01,
          "seconds_left": 10.0, "p95_ms": {"warp": 3.0, "segment": 40.0}, "model": "accurate",
          "top_candidates": [{"id": "r1", "score": 0.01, "entropy": 0.4, "level": 1, "prior": 0.9}],
          "counts_by_class": {"crack": 1}, "critical_pending": 0}


def test_validator_accepts_allow_listed_moves():
    ok = P.validate({"strategy": "switch_model", "params": {"model": "fp16"}, "rationale": "p95 high"}, DIGEST)
    assert ok == {"strategy": "switch_model", "params": {"model": "fp16"}, "rationale": "p95 high"}
    ok = P.validate({"strategy": "change_level", "params": {"level": 0, "region_id": "r1"}, "rationale": ""}, DIGEST)
    assert ok["params"] == {"level": 0, "region_id": "r1"}
    assert P.validate({"strategy": "stop", "rationale": "done"}, DIGEST)["strategy"] == "stop"


@pytest.mark.parametrize("bad", [
    {"strategy": "shell", "params": {"cmd": "ls"}, "rationale": ""},
    {"strategy": "switch_model", "params": {"model": "gpt-9"}, "rationale": ""},
    {"strategy": "change_level", "params": {"level": 7, "region_id": "r1"}, "rationale": ""},
    {"strategy": "change_level", "params": {"level": True, "region_id": "r1"}, "rationale": ""},
    {"strategy": "reverify", "params": {"region_id": "r99"}, "rationale": ""},
    {"strategy": "continue", "params": {}, "rationale": "x" * 201},
    {"strategy": "continue", "params": "nope", "rationale": ""},
    ["continue"],
])
def test_validator_rejects(bad):
    with pytest.raises(P.Invalid):
        P.validate(bad, DIGEST)


def test_budget_adding_move_needs_budget():
    with pytest.raises(P.Invalid, match="budget"):
        P.validate({"strategy": "reverify", "params": {"region_id": "r1"}, "rationale": ""},
                   {**DIGEST, "looks_left": 0})


def test_20_timeouts_and_20_malformed_all_recover():
    results = [P.consult(P.ChaosPlanner("timeout"), DIGEST, timeout_s=0.05) for _ in range(20)]
    results += [P.consult(P.ChaosPlanner("malformed"), DIGEST) for _ in range(20)]
    results += [P.consult(lambda d: "not json at all", DIGEST) for _ in range(5)]
    results += [P.consult(lambda d: 1 / 0, DIGEST) for _ in range(5)]
    assert all(proposal is None for proposal, _, _ in results)
    assert [s for _, s, _ in results[:20]] == ["fallback"] * 20
    assert [s for _, s, _ in results[20:40]] == ["rejected"] * 20


def test_planner_sees_numbers_and_fixed_names_only():
    allowed = {"accurate", "fast", "fp16", "r1", "crack", "spalling", "efflorescence", "exposed_rebar", "corrosion"}

    def strings(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from strings(v)
        elif isinstance(o, list):
            for v in o:
                yield from strings(v)
        elif isinstance(o, str):
            yield o

    assert set(strings(DIGEST)) <= allowed
    json.dumps(DIGEST)


def test_http_planner_url_policy(monkeypatch):
    with pytest.raises(ValueError):
        P.HttpPlanner("http://169.254.169.254/latest")  # cloud metadata over plain http
    P.HttpPlanner("http://127.0.0.1:8080/v1")
    P.HttpPlanner("https://api.groq.com/openai/v1")
    monkeypatch.delenv("DRISHYAM_PLANNER_URL", raising=False)
    assert P.from_env() is None


def stats_with(op, values):
    s = G.RuntimeStats()
    for v in values:
        s.record(op, v)
    return s


def budget(**kw):
    return G.Budget(**{"pixels_cap": 1_000_000, "usd_cap": 0.05, "seconds_cap": 20.0, **kw})


def test_governor_halts_on_kill_and_caps(monkeypatch):
    monkeypatch.delenv("DRISHYAM_KILL_SWITCH", raising=False)
    s, st = G.RuntimeStats(), G.Settings()
    assert G.check(budget(), s, st, 1000, 500, kill=True).cause == "kill-switch"
    assert G.check(budget(pixels_used=999_500), s, st, 1000, 500, kill=False).cause == "cap-hit"
    assert G.check(budget(seconds_used=20.0), s, st, 1000, 500, kill=False).cause == "cap-hit"
    b = budget(usd_cap=0.0001, seconds_used=2.0)  # 2 s at $0.339/h is $0.00019
    assert G.check(b, s, st, 1000, 500, kill=False).cause == "cap-hit"
    monkeypatch.setenv("DRISHYAM_KILL_SWITCH", "1")
    assert G.check(budget(), s, st, 1000, 500, kill=False).halt


def test_global_kill_file(tmp_path, monkeypatch):
    flag = tmp_path / "KILL"
    monkeypatch.setenv("DRISHYAM_KILL_FILE", str(flag))
    assert not G.global_kill()
    flag.touch()
    assert G.global_kill()


def test_p95_breach_switches_model_then_level():
    st = G.Settings()
    s = stats_with("segment:accurate", [40, 40, 900])
    v = G.check(budget(), s, st, 1000, 500, kill=False)
    assert (v.cause, st.model, st.level_bias) == ("p95>SLA", "fast", 0)
    s.record("segment:fast", 800)
    v = G.check(budget(), s, st, 1000, 500, kill=False)
    assert (v.cause, st.model, st.level_bias) == ("p95>SLA", "fast", 1)
    assert not s.samples["segment:fast"]  # window reset so the next decision re-measures


def test_warp_spike_needs_a_real_spike():
    st = G.Settings()
    assert G.check(budget(), stats_with("warp", [1, 1, 1, 1, 3]), st, 1000, 500, kill=False).cause is None
    v = G.check(budget(), stats_with("warp", [2, 2, 2, 2, 2, 2, 2, 2, 2, 60]), st, 1000, 500, kill=False)
    assert v.cause == "warp-spike" and st.size == 384 and st.batch


def test_usd_80_percent_switches_to_critical_only():
    st = G.Settings()
    b = budget(usd_cap=0.001, seconds_used=0.81 * 0.001 * 3600 / G.USD_PER_HOUR)
    assert G.check(b, G.RuntimeStats(), st, 1000, 500, kill=False).cause == "usd≥80%"
    assert st.critical_only


def test_http_planner_rejects_lookalike_loopback_hosts():
    for bad in ("http://localhost.evil.com/v1", "http://127.0.0.1.evil.com/v1", "ftp://127.0.0.1/", "https:///x"):
        with pytest.raises(ValueError):
            P.HttpPlanner(bad)
    P.HttpPlanner("http://[::1]:8080/v1")
