"""Tests for the R4 rescue router and public threshold selection."""

from __future__ import annotations

import pytest

from xh202615.metrics import cer_stats
from xh202615.r4_rescue import (
    POLICY_MAX_FRR,
    POLICY_MIN_RR,
    public_baselines,
    rescue_metrics,
    rescue_text,
    safe_rescue_text,
    select_rescue_threshold,
)


# --- rescue_text -----------------------------------------------------------


def test_safe_rescue_uses_raw_only_when_fusion_accepted_and_present():
    # head present + fusion non-empty + raw -> raw
    assert safe_rescue_text(0.9, "raw", "fusion", threshold=0.5) == ("raw", "safe_raw")


def test_safe_rescue_keeps_fusion_when_head_absent():
    assert safe_rescue_text(0.1, "raw", "fusion", threshold=0.5) == ("fusion", "fusion")


def test_safe_rescue_never_rescues_fusion_rejection():
    # fusion rejected (empty) + head present -> stays empty (no new false accept)
    assert safe_rescue_text(0.99, "raw", "", threshold=0.5) == ("", "fusion_reject")


def test_safe_rescue_empty_raw_keeps_fusion():
    assert safe_rescue_text(0.9, "", "fusion", threshold=0.5) == ("fusion", "fusion")


def test_safe_rescue_preserves_rejection_rate():
    # RR over absent rows is identical to the fallback regardless of threshold,
    # because fusion-rejected absent rows are never rescued.
    flags = [1, 1, 0, 0, 0]
    refs = ["打开空调", "关灯", "", "", ""]
    raw = ["打开空调", "关灯", "噪声", "噪声", "噪声"]
    fall = ["打开空调", "", "", "", "噪声"]  # fusion rejects 2 absent, accepts 1
    base = rescue_metrics(flags, refs, fall)
    for thr in [0.0, 0.3, 0.5, 0.8, 1.0]:
        outs = [safe_rescue_text(p, r, f, threshold=thr)[0] for p, r, f in zip([0.9, 0.1, 0.9, 0.1, 0.9], raw, fall)]
        m = rescue_metrics(flags, refs, outs)
        assert m["avg_rr"] == base["avg_rr"], f"RR changed at thr={thr}"
        assert m["false_reject_rate"] == base["false_reject_rate"], f"FRR changed at thr={thr}"


# --- rescue_text -----------------------------------------------------------


def test_rescue_text_present_uses_raw():
    assert rescue_text(0.9, "打开空调", "", threshold=0.5) == ("打开空调", "raw")


def test_rescue_text_absent_uses_fallback():
    assert rescue_text(0.1, "打开空调", "fallback", threshold=0.5) == ("fallback", "fallback")


def test_rescue_text_empty_raw_falls_back_even_if_present():
    # present decision but no raw transcript -> fallback (cannot fabricate text)
    assert rescue_text(0.9, "", "fallback", threshold=0.5) == ("fallback", "fallback_empty_raw")


def test_rescue_text_threshold_boundary():
    # >= threshold is present (inclusive)
    assert rescue_text(0.5, "raw", "fall", threshold=0.5)[1] == "raw"
    assert rescue_text(0.4999, "raw", "fall", threshold=0.5)[1] == "fallback"


def test_rescue_text_rejects_non_finite():
    with pytest.raises(ValueError):
        rescue_text(float("nan"), "raw", "fall", threshold=0.5)


# --- rescue_metrics --------------------------------------------------------


def test_rescue_metrics_perfect_raw_perfect_reject():
    # 2 present rows perfectly transcribed, 2 absent rows rejected (empty)
    flags = [1, 1, 0, 0]
    refs = ["打开空调", "关灯", "", ""]
    outs = ["打开空调", "关灯", "", ""]
    m = rescue_metrics(flags, refs, outs)
    assert m["avg_cer"] == 0.0
    assert m["avg_rr"] == 1.0
    assert m["overall"] == 1.0
    assert m["false_reject_rate"] == 0.0
    assert m["reject_accuracy"] == 1.0
    assert m["pos_count"] == 2 and m["neg_count"] == 2


def test_rescue_metrics_rejected_present_is_full_deletion():
    # present row rejected (empty) -> all ref chars become deletions
    ref = "打开空调"
    expected = cer_stats(ref, "")
    m = rescue_metrics([1, 0], [ref, ""], ["", ""])
    assert m["avg_cer"] == pytest.approx(expected.errors / expected.ref_chars)
    assert m["deletions"] == expected.deletions
    assert m["false_reject_rate"] == 1.0  # the one present row was rejected


def test_rescue_metrics_false_accept_lowers_rr():
    # absent row transcribed (non-empty) -> false accept
    m = rescue_metrics([1, 0], ["打开空调", ""], ["打开空调", "噪声文本"])
    assert m["avg_rr"] == 0.0
    assert m["reject_accuracy"] == 0.0
    assert m["false_reject_rate"] == 0.0


def test_rescue_metrics_overall_formula():
    # Overall = ((1 - CER) + RR) / 2
    m = rescue_metrics([1, 1, 0, 0], ["打开空调", "关闭灯光", "", ""], ["打开空调", "关闭灯", "", ""])
    expected = ((1.0 - m["avg_cer"]) + m["avg_rr"]) / 2.0
    assert m["overall"] == pytest.approx(expected)


def test_rescue_metrics_validates_lengths_and_flags():
    with pytest.raises(ValueError):
        rescue_metrics([1, 0], ["a"], ["b"])
    with pytest.raises(ValueError):
        rescue_metrics([1, 2], ["a", "b"], ["c", "d"])


# --- select_rescue_threshold ----------------------------------------------


def _separable_case():
    # present rows have high probability, absent rows low -> separable
    present = [1, 1, 1, 0, 0, 0]
    probs = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05]
    refs = ["打开空调", "关闭灯光", "调高音量", "", "", ""]
    raw = ["打开空调", "关闭灯光", "调高音量", "噪声", "噪声", "噪声"]  # raw transcribes absent too
    fall = [""] * 6  # public stand-in for fusion = empty
    return present, probs, refs, raw, fall


def test_select_rescue_threshold_separable_meets_policy():
    present, probs, refs, raw, fall = _separable_case()
    result = select_rescue_threshold(present, probs, refs, raw, fall)
    thr = result["threshold"]
    assert result["threshold_source"] == "public_validation"
    assert result["metrics"]["false_reject_rate"] <= POLICY_MAX_FRR
    assert result["metrics"]["reject_accuracy"] >= POLICY_MIN_RR
    # threshold must sit between the absent (<=0.2) and present (>=0.7) masses;
    # 0.7 itself is an eligible candidate (present 0.7 >= 0.7, absent 0.2 < 0.7)
    # and is the highest threshold achieving Overall 1.0, so it is selected.
    assert 0.2 < thr <= 0.7
    assert result["num_eligible"] >= 1


def test_select_rescue_threshold_maximizes_overall():
    present, probs, refs, raw, fall = _separable_case()
    result = select_rescue_threshold(present, probs, refs, raw, fall)
    # Any eligible threshold in (0.2, 0.7] yields perfect separation -> Overall 1.0
    assert result["metrics"]["overall"] == pytest.approx(1.0)
    assert result["metrics"]["avg_cer"] == 0.0
    assert result["metrics"]["avg_rr"] == 1.0


def test_select_rescue_threshold_fail_closed_when_inseparable():
    # present and absent probabilities fully overlap -> no threshold meets policy
    present = [1, 1, 0, 0]
    probs = [0.4, 0.6, 0.5, 0.55]
    refs = ["打开空调", "关闭灯光", "", ""]
    raw = ["打开空调", "关闭灯光", "噪声", "噪声"]
    fall = ["", "", "", ""]
    with pytest.raises(ValueError, match="no eligible public rescue threshold"):
        select_rescue_threshold(present, probs, refs, raw, fall)


def test_select_rescue_threshold_policy_too_strict_fails_closed():
    # Separable enough for the default policy (one hard present at 0.3, one hard
    # absent at 0.8): at thr=0.5, FRR=1/10=0.10 (<=0.10) and RR=9/10=0.90 (>=0.85),
    # so the default policy is satisfied. But a perfect policy (FRR=0 AND RR=1)
    # is impossible because the 0.3-present and 0.8-absent overlap -> fail-closed.
    present = [1] * 10 + [0] * 10
    probs = [0.9] * 9 + [0.3] + [0.8] + [0.1] * 9
    refs = ["打开空调"] * 10 + [""] * 10
    raw = ["打开空调"] * 10 + ["噪声"] * 10
    fall = [""] * 20
    # default policy succeeds
    result = select_rescue_threshold(present, probs, refs, raw, fall)
    assert result["metrics"]["overall"] >= 0.0
    # perfect policy cannot be satisfied -> fail-closed
    with pytest.raises(ValueError):
        select_rescue_threshold(present, probs, refs, raw, fall, max_frr=0.0, min_rr=1.0)


def test_select_rescue_threshold_requires_both_classes():
    present = [1, 1]
    probs = [0.9, 0.8]
    with pytest.raises(ValueError):
        select_rescue_threshold(present, probs, ["a", "b"], ["x", "y"], ["", ""])


# --- public_baselines -----------------------------------------------------


def test_public_baselines_raw_alone_and_fallback_only():
    present, probs, refs, raw, fall = _separable_case()
    base = public_baselines(present, refs, raw, fall)
    # raw_alone: every sample -> raw transcript. CER over present = 0 (perfect),
    # RR over absent = 0 (raw transcribes absent -> false accepts).
    assert base["raw_alone"]["avg_cer"] == 0.0
    assert base["raw_alone"]["avg_rr"] == 0.0
    assert base["raw_alone"]["overall"] == 0.5
    # fallback_only (empty): CER over present = 1.0 (full deletions), RR = 1.0.
    assert base["fallback_only"]["avg_cer"] == 1.0
    assert base["fallback_only"]["avg_rr"] == 1.0
    assert base["fallback_only"]["overall"] == 0.5


def test_public_baselines_rescue_beats_both_corners():
    present, probs, refs, raw, fall = _separable_case()
    base = public_baselines(present, refs, raw, fall)
    result = select_rescue_threshold(present, probs, refs, raw, fall)
    rescue_overall = result["metrics"]["overall"]
    assert rescue_overall > base["raw_alone"]["overall"]
    assert rescue_overall > base["fallback_only"]["overall"]
