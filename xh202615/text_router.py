"""Text-aware rejection helpers for V2 experiments."""

from __future__ import annotations

from dataclasses import dataclass

from .backends import SpeakerScores
from .postprocess import ACTION_TERMS, DEVICE_TERMS, SETTING_TERMS
from .text import normalize_text


MEDIA_TERMS = (
    "播放",
    "暂停",
    "继续",
    "上一首",
    "下一首",
    "音乐",
    "歌曲",
    "歌",
    "专辑",
    "电影",
    "电视剧",
    "电台",
    "新闻",
    "儿歌",
    "故事",
    "相声",
    "有声书",
)

LIFE_TERMS = (
    "出门",
    "回家",
    "到家",
    "吃饭",
    "睡觉",
    "起床",
    "做饭",
    "洗澡",
    "休息",
    "上班",
    "下班",
    "通勤",
    "运动",
    "健身",
)

QA_TERMS = (
    "什么",
    "怎么",
    "如何",
    "为什么",
    "多少",
    "几",
    "几点",
    "叫什么",
    "推荐",
    "查询",
    "搜索",
    "告诉我",
    "帮我查",
)

TOOL_TERMS = (
    "闹钟",
    "提醒",
    "定时",
    "倒计时",
    "日程",
    "预约",
    "取消",
    "备忘",
    "计时",
    "导航",
    "路线",
    "天气",
)


@dataclass(frozen=True)
class TextEvidence:
    normalized_text: str
    text_length: int
    domain_hits: int
    action_hits: int
    device_hits: int
    setting_hits: int
    media_hits: int = 0
    life_hits: int = 0
    qa_hits: int = 0
    tool_hits: int = 0

    @property
    def domain_score(self) -> int:
        return (
            self.action_hits
            + self.device_hits
            + self.setting_hits
            + self.media_hits
            + self.life_hits
            + self.qa_hits
            + self.tool_hits
        )

    @property
    def assistant_intent_score(self) -> int:
        return self.media_hits + self.life_hits + self.qa_hits + self.tool_hits


@dataclass(frozen=True)
class TextRouteResult:
    reject: bool
    reason: str


def analyze_text(text: str | None) -> TextEvidence:
    normalized = normalize_text(text)

    def count_hits(terms: tuple[str, ...]) -> int:
        return sum(1 for term in terms if term in normalized)

    action_hits = count_hits(ACTION_TERMS)
    device_hits = count_hits(DEVICE_TERMS)
    setting_hits = count_hits(SETTING_TERMS)
    media_hits = count_hits(MEDIA_TERMS)
    life_hits = count_hits(LIFE_TERMS)
    qa_hits = count_hits(QA_TERMS)
    tool_hits = count_hits(TOOL_TERMS)
    all_terms = ACTION_TERMS + DEVICE_TERMS + SETTING_TERMS + MEDIA_TERMS + LIFE_TERMS + QA_TERMS + TOOL_TERMS
    domain_hits = count_hits(all_terms)
    return TextEvidence(
        normalized_text=normalized,
        text_length=len(normalized),
        domain_hits=domain_hits,
        action_hits=action_hits,
        device_hits=device_hits,
        setting_hits=setting_hits,
        media_hits=media_hits,
        life_hits=life_hits,
        qa_hits=qa_hits,
        tool_hits=tool_hits,
    )


def _leq(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def route_by_text(text: str | None, scores: SpeakerScores, config: dict) -> TextRouteResult:
    text_router = config.get("text_router", {})
    if not text_router.get("enabled", False):
        return TextRouteResult(False, "text_router_disabled")

    reject_cfg = text_router.get("reject", {})
    evidence = analyze_text(text)
    if evidence.text_length < int(reject_cfg.get("min_text_length", 10)):
        return TextRouteResult(False, f"text_too_short={evidence.text_length}")

    max_domain_score = int(reject_cfg.get("max_domain_score", 0))
    if evidence.domain_score > max_domain_score:
        return TextRouteResult(False, f"domain_score={evidence.domain_score}")

    sim_max = reject_cfg.get("speaker_similarity_max")
    prob_max = reject_cfg.get("target_probability_max")
    speaker_votes = 0
    if sim_max is not None and _leq(scores.global_similarity, float(sim_max)):
        speaker_votes += 1
    if sim_max is not None and _leq(scores.topk_similarity, float(sim_max)):
        speaker_votes += 1
    if prob_max is not None and _leq(scores.target_probability, float(prob_max)):
        speaker_votes += 1

    min_speaker_votes = int(reject_cfg.get("min_speaker_votes", 2))
    if speaker_votes < min_speaker_votes:
        return TextRouteResult(False, f"text_non_domain_but_speaker_votes={speaker_votes}")

    return TextRouteResult(
        True,
        (
            "text_non_domain_reject"
            f":len={evidence.text_length},domain_score={evidence.domain_score},speaker_votes={speaker_votes}"
        ),
    )
