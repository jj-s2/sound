"""Command-aware ASR correction for V3 experiments."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from .postprocess import ACTION_TERMS, DEVICE_TERMS, SETTING_TERMS, has_command_context
from .text import clean_asr_text, normalize_text


COMMAND_PHRASES = (
    ACTION_TERMS
    + DEVICE_TERMS
    + SETTING_TERMS
    + (
        "风向",
        "风摆",
        "出风口",
        "防直吹",
        "无风感",
        "左右风",
        "上下风",
        "左右摆风",
        "自动模式",
        "睡眠模式",
        "省电模式",
        "节能模式",
        "ECO模式",
        "抽湿",
        "抽湿模式",
        "智控温",
        "自清洁",
        "洗烘套装",
        "干衣机",
        "最大",
        "最小",
        "最低",
        "最高",
        "上边",
        "下边",
        "左边",
        "右边",
        "百分之",
        "二十",
        "三十",
        "四十",
        "五十",
        "六十",
        "七十",
        "八十",
        "九十",
    )
)


_CONFUSION_MAP = {
    "丰": "风",
    "分": "风",
    "放": "风",
    "方": "风",
    "房": "防",
    "纺": "防",
    "止": "直",
    "织": "直",
    "追": "吹",
    "本": "摆",
    "板": "摆",
    "速": "速",
    "数": "速",
    "富": "速",
    "量": "量",
    "梁": "量",
    "良": "量",
    "调": "调",
    "掉": "调",
    "到": "到",
    "道": "到",
    "导": "到",
    "自": "制",
    "智": "制",
    "至": "制",
    "之": "制",
    "致": "制",
    "热": "热",
    "惹": "热",
    "冷": "冷",
    "领": "冷",
    "湿": "湿",
    "室": "湿",
    "抽": "除",
    "插": "湿",
    "屏": "屏",
    "平": "屏",
    "瓶": "屏",
    "幕": "幕",
    "目": "幕",
    "睡": "睡",
    "水": "睡",
    "眠": "眠",
    "棉": "眠",
    "模": "模",
    "膜": "模",
    "式": "式",
    "试": "式",
    "氏": "式",
    "实": "式",
    "省": "省",
    "神": "省",
    "电": "电",
    "店": "电",
    "窗": "窗",
    "仓": "窗",
    "创": "窗",
    "帘": "帘",
    "联": "帘",
    "面": "帘",
    "烟": "烟",
    "颜": "烟",
    "严": "烟",
    "言": "烟",
    "机": "机",
    "接": "机",
    "鸡": "机",
    "升": "升",
    "生": "升",
    "声": "升",
    "申": "升",
    "请": "起",
    "起": "起",
    "空": "空",
    "控": "空",
    "烘": "烘",
    "轰": "烘",
    "公": "烘",
    "干": "干",
    "感": "感",
    "甘": "感",
    "无": "无",
    "五": "无",
    "扫": "扫",
    "草": "扫",
}


@dataclass(frozen=True)
class CommandCorrectionResult:
    text: str
    changed: bool
    reason: str


def _encoded(text: str) -> str:
    return "".join(_CONFUSION_MAP.get(ch, ch) for ch in normalize_text(text))


_PHRASE_DATA = tuple((phrase, normalize_text(phrase), _encoded(phrase)) for phrase in COMMAND_PHRASES)


def _ratio(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    raw = SequenceMatcher(None, left_norm, right_norm).ratio()
    encoded = SequenceMatcher(None, _encoded(left_norm), _encoded(right_norm)).ratio()
    return max(raw, encoded)


def _apply_fixed_rules(text: str) -> tuple[str, list[str]]:
    rules = (
        ("关节空调", "关机空调"),
        ("康夏空调", "关下空调"),
        ("空角空调", "关掉空调"),
        ("感谢空调", "开开空调"),
        ("丰富六十", "风速六十"),
        ("放手给到自动", "风速调到自动"),
        ("分数调到最大分", "风速调到最大风"),
        ("防止吹", "防直吹"),
        ("纺织吹", "防直吹"),
        ("开房直追", "开防直吹"),
        ("五风干", "无风感"),
        ("五烘干", "无风感"),
        ("打开草房", "打开扫风"),
        ("关闭神电模式", "关闭省电模式"),
        ("关闭生活模式", "关闭ECO模式"),
        ("打开一些模式", "打开ECO模式"),
        ("开启关抽插模式", "开启抽湿模式"),
        ("开启之不温", "开启智控温"),
        ("风量量小", "风量最小"),
        ("风速调高声", "风速调大"),
        ("关点睡眠", "关掉睡眠"),
        ("烟机申请", "烟机升起"),
        ("灯光无烟机", "厨房灯光烟机"),
    )
    updated = text
    reasons = []
    for old, new in rules:
        if old not in updated:
            continue
        next_text = updated.replace(old, new)
        if next_text != updated:
            updated = next_text
            reasons.append(f"{old}->{new}")
    return updated, reasons


def _has_exact_phrase(span_norm: str) -> bool:
    return any(len(phrase_norm) >= 2 and phrase_norm in span_norm for _, phrase_norm, _ in _PHRASE_DATA)


def _best_phrase_replacement(span: str, existing_norm: str, min_score: float) -> tuple[str | None, float]:
    best_phrase = None
    best_score = 0.0
    span_norm = normalize_text(span)
    if _has_exact_phrase(span_norm):
        return None, 0.0
    span_encoded = _encoded(span_norm)
    for phrase, phrase_norm, phrase_encoded in _PHRASE_DATA:
        if span_norm == phrase_norm:
            continue
        if phrase_norm in existing_norm:
            continue
        raw = SequenceMatcher(None, span_norm, phrase_norm).ratio()
        encoded = SequenceMatcher(None, span_encoded, phrase_encoded).ratio()
        score = max(raw, encoded)
        if score > best_score:
            best_phrase = phrase
            best_score = score
    if best_score >= min_score:
        return best_phrase, best_score
    return None, best_score


def _apply_phrase_corrections(text: str, min_score: float, max_changes: int) -> tuple[str, list[str]]:
    normalized = normalize_text(text)
    if not normalized:
        return text, []

    chars = list(text)
    existing_norm = normalize_text(text)
    changes = []
    idx = 0
    while idx < len(chars) and len(changes) < max_changes:
        best = None
        best_end = idx
        best_score = 0.0
        for span_len in range(4, 1, -1):
            end = idx + span_len
            if end > len(chars):
                continue
            span = "".join(chars[idx:end])
            phrase, score = _best_phrase_replacement(span, existing_norm, min_score)
            if phrase and score > best_score:
                best = phrase
                best_end = end
                best_score = score
        if best is None:
            idx += 1
            continue
        old = "".join(chars[idx:best_end])
        chars[idx:best_end] = list(best)
        changes.append(f"{old}->{best}:{best_score:.2f}")
        idx += len(best)
    return "".join(chars), changes


def correct_command_text(text: str | None, config: dict | None = None) -> CommandCorrectionResult:
    """Correct short command-like ASR text using a domain phrase lexicon."""

    cfg = config or {}
    cleaned = clean_asr_text(text)
    if not cfg.get("enabled", False):
        return CommandCorrectionResult(cleaned, False, "disabled")

    normalized = normalize_text(cleaned)
    short_limit = int(cfg.get("short_text_max_length", 8))
    if len(normalized) > short_limit and not has_command_context(cleaned):
        return CommandCorrectionResult(cleaned, False, "no_command_context")

    updated, reasons = _apply_fixed_rules(cleaned)
    max_phrase_text_length = int(cfg.get("max_phrase_text_length", 16))
    if len(normalize_text(updated)) <= max_phrase_text_length:
        min_score = float(cfg.get("min_similarity", 0.88))
        max_changes = int(cfg.get("max_phrase_changes", 1))
        updated, phrase_reasons = _apply_phrase_corrections(updated, min_score, max_changes)
        reasons.extend(phrase_reasons)

    if not reasons:
        return CommandCorrectionResult(updated, False, "no_change")
    return CommandCorrectionResult(updated, True, ";".join(reasons))
