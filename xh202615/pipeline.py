"""Inference pipeline for V0-V3 experiments."""

from __future__ import annotations

import time  #统计单条样本推理耗时。
from dataclasses import asdict, dataclass #定义结构化预测结果，并转成 JSON。

from .audio_features import read_wav_info #读取音频时长等信息。
from .backends import ScoreCsvSpeakerBackend, make_asr_backend  #创建 ASR 后端，现在可以是 no-op 或外部 ASR 文件。
from .command_corrector import correct_command_text
from .postprocess import postprocess_asr_text
from .router import route_sample  #根据声纹分数和配置决定走哪条路线。
from .text_router import route_by_text


@dataclass(frozen=True)
class Prediction:
    id: str
    recognition_text: str
    route: str
    route_reason: str
    raw_asr_text: str
    enhanced_asr_text: str | None
    asr_backend: str
    speaker_backend: str
    latency_ms: float
    command_duration_sec: float

    def to_json(self) -> dict:
        return asdict(self)


class Pipeline:
    def __init__(self, config: dict, asr_map: str | None = None, speaker_scores: str | None = None):
        self.config = config
        self.asr = make_asr_backend(asr_map)
        self.speaker = ScoreCsvSpeakerBackend(speaker_scores)

    def infer(self, sample) -> Prediction:
        start = time.perf_counter()  #记录开始时间
        if self.config.get("audio_probe", {}).get("enabled", False):
            info = read_wav_info(sample.command_audio)
            duration_sec = info.duration_sec
        else:
            duration_sec = 0.0
        scores = self.speaker.score(sample)
        route = route_sample(scores, self.config)

        if route.route == "reject":   #如果是拒识，说明认为不是主人，不跑asr
            raw_asr_text = ""
            text = ""
            asr_backend = "skipped"
            enhanced_asr_text = None
        else:
            raw = self.asr.transcribe(sample)
            raw_asr_text = raw.text
            enhanced = postprocess_asr_text(raw.text, self.config.get("postprocess", {}))
            text = enhanced.text
            enhanced_asr_text = enhanced.text if enhanced.changed else None
            corrected = correct_command_text(text, self.config.get("command_corrector", {}))
            text = corrected.text
            if corrected.changed:
                enhanced_asr_text = corrected.text
            asr_backend = raw.backend
            text_route = route_by_text(text, scores, self.config)
            if text_route.reject:
                text = ""
                route = type(route)("reject", text_route.reason)

        total_ms = (time.perf_counter() - start) * 1000
        return Prediction(
            id=str(sample.id),
            recognition_text=text,
            route=route.route,
            route_reason=route.reason,
            raw_asr_text=raw_asr_text,
            enhanced_asr_text=enhanced_asr_text,
            asr_backend=asr_backend,
            speaker_backend=scores.backend,
            latency_ms=total_ms,
            command_duration_sec=duration_sec,
        )
