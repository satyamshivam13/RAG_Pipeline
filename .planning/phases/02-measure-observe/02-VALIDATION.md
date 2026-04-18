---
phase: 02
slug: measure-observe
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-04-18
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest |
| **Config file** | tests/conftest.py |
| **Quick run command** | `pytest tests/test_eval_dataset.py tests/test_telemetry.py -q` |
| **Full suite command** | `pytest tests -q` |
| **Estimated runtime** | ~90 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_eval_dataset.py tests/test_telemetry.py -q`
- **After every plan wave:** Run `pytest tests -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | MEAS-01 | T-02-01 | Loader rejects malformed dataset rows | unit | `pytest tests/test_eval_dataset.py -q` | ✅ | ⬜ pending |
| 02-01-02 | 01 | 1 | MEAS-02 | T-02-02 | Eval runner outputs required metric keys | unit | `pytest tests/test_eval_runner.py -q` | ✅ | ⬜ pending |
| 02-02-01 | 02 | 1 | MEAS-03 | T-02-05 | Logs carry correlation_id without raw prompt dumps | unit | `pytest tests/test_telemetry.py -q` | ✅ | ⬜ pending |
| 02-02-02 | 02 | 1 | MEAS-04 | T-02-06 | Root/child span model present across stages | unit | `pytest tests/test_telemetry.py -q` | ✅ | ⬜ pending |
| 02-03-01 | 03 | 2 | MEAS-02 | T-02-09 | Gate parser fails safely on missing/invalid metrics | unit | `pytest tests/test_quality_gates.py -q` | ✅ | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_eval_dataset.py` — dataset schema and loader contracts
- [ ] `tests/test_eval_runner.py` — eval runner/report schema checks
- [ ] `tests/test_telemetry.py` — correlation/span propagation checks
- [ ] `tests/test_quality_gates.py` — threshold gate behavior checks

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Console telemetry readability under real query load | MEAS-03 | Human judgment on observability usefulness | Run `python demo.py`, execute query, confirm correlation_id and stage latencies are visible and interpretable |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 120s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
