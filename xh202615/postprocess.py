"""Conservative ASR text post-processing for smart-home commands."""

from __future__ import annotations

from dataclasses import dataclass

from .text import clean_asr_text, normalize_text


ACTION_TERMS = (
    "打开",
    "开启",
    "关闭",
    "关掉",
    "设置",
    "设为",
    "设定",
    "切换",
    "调整",
    "调到",
    "调成",
    "调至",
    "调高",
    "调低",
    "调大",
    "调小",
    "调亮",
    "调暗",
    "开到",
    "打到",
    "暂停",
    "播放",
    "预约",
    "取消",
)

DEVICE_TERMS = (
    "空调",
    "灯光",
    "灯",
    "窗帘",
    "纱帘",
    "洗衣机",
    "洗碗机",
    "烟机",
    "烤箱",
    "冰箱",
    "新风",
    "热水器",
    "电视",
    "屏幕",
    "音乐",
)

SETTING_TERMS = (
    "风速",
    "风量",
    "温度",
    "亮度",
    "色温",
    "制冷",
    "制热",
    "除湿",
    "模式",
    "净化",
    "清洁",
    "烘干",
    "洗护",
    "下降",
    "上升",
)


@dataclass(frozen=True)
class PostprocessResult:
    text: str
    changed: bool
    reason: str


def has_command_context(text: str) -> bool:
    normalized = normalize_text(text)
    terms = ACTION_TERMS + DEVICE_TERMS + SETTING_TERMS
    return any(term in normalized for term in terms)


def postprocess_asr_text(text: str | None, config: dict | None = None) -> PostprocessResult:
    """Apply small, domain-safe corrections to ASR output."""

    cfg = config or {}
    cleaned = clean_asr_text(text)
    if not cfg.get("enabled", False):
        return PostprocessResult(cleaned, False, "disabled")

    context_required = cfg.get("domain_context_required", True)
    if context_required and not has_command_context(cleaned):
        return PostprocessResult(cleaned, False, "no_command_context")

    replacements = (
        ("颜接", "烟机"),
        ("烟接", "烟机"),
        ("仓帘", "窗帘"),
        ("窗联", "窗帘"),
        ("创联", "窗帘"),
        ("上来", "纱帘"),
        ("时机", "洗衣机"),
        ("洗一机", "洗衣机"),
        ("洗手机", "洗衣机"),
        ("洗完机", "洗碗机"),
        ("洗碗器", "洗碗机"),
        ("自热", "制热"),
        ("之热", "制热"),
        ("致热", "制热"),
        ("公单", "烘干"),
        ("风干", "烘干"),
        ("开起", "开启"),
        ("管掉", "关掉"),
        ("智清洁", "自清洁"),
    )

    updated = cleaned
    changed_terms = []
    for old, new in replacements:
        if old not in updated:
            continue
        next_text = updated.replace(old, new)
        if next_text != updated:
            updated = next_text
            changed_terms.append(f"{old}->{new}")

    if not changed_terms:
        return PostprocessResult(updated, False, "no_change")
    return PostprocessResult(updated, True, ";".join(changed_terms))
