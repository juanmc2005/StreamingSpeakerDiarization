from typing import Optional, List, Any

import rx
import rx.operators as ops
import torch

from . import blocks
from . import models as m
from . import operators as dops
from . import sources as src
from . import utils


class PipelineConfig:
    def __init__(
        self,
        segmentation: Optional[m.SegmentationModel] = None,
        embedding: Optional[m.EmbeddingModel] = None,
        duration: Optional[float] = None,
        step: float = 0.5,
        latency: Optional[float] = None,
        tau_active: float = 0.6,
        rho_update: float = 0.3,
        delta_new: float = 1,
        gamma: float = 3,
        beta: float = 10,
        max_speakers: int = 20,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        # Default segmentation model is pyannote/segmentation
        self.segmentation = segmentation
        if self.segmentation is None:
            self.segmentation = m.SegmentationModel.from_pyannote("pyannote/segmentation")

        # Default duration is the one given by the segmentation model
        self.duration = duration
        if self.duration is None:
            self.duration = self.segmentation.get_duration()

        # Expected sample rate is given by the segmentation model
        self.sample_rate = self.segmentation.get_sample_rate()

        # Default embedding model is pyannote/embedding
        self.embedding = embedding
        if self.embedding is None:
            self.embedding = m.EmbeddingModel.from_pyannote("pyannote/embedding")

        # Latency defaults to the step duration
        self.step = step
        self.latency = latency
        if self.latency is None:
            self.latency = self.step

        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.gamma = gamma
        self.beta = beta
        self.max_speakers = max_speakers

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def from_namespace(args: Any) -> 'PipelineConfig':
        return PipelineConfig(
            segmentation=getattr(args, "segmentation", None),
            embedding=getattr(args, "embedding", None),
            duration=getattr(args, "duration", None),
            step=args.step,
            latency=args.latency,
            tau_active=args.tau,
            rho_update=args.rho,
            delta_new=args.delta,
            gamma=args.gamma,
            beta=args.beta,
            max_speakers=args.max_speakers,
            device=args.device,
        )

    def last_chunk_end_time(self, conv_duration: float) -> Optional[float]:
        """Return the end time of the last chunk for a given conversation duration.

        Parameters
        ----------
        conv_duration: float
            Duration of a conversation in seconds.
        """
        return conv_duration - conv_duration % self.step


class OnlineSpeakerTracking:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def get_end_time(self, duration: Optional[float]) -> Optional[float]:
        return None if duration is None else self.config.last_chunk_end_time(duration)

    def get_operators(self, source: src.AudioSource) -> List[dops.Operator]:
        clustering = blocks.OnlineSpeakerClustering(
            self.config.tau_active,
            self.config.rho_update,
            self.config.delta_new,
            "cosine",
            self.config.max_speakers,
        )
        end_time = self.get_end_time(source.duration)
        pred_aggregation = blocks.DelayedAggregation(
            self.config.step, self.config.latency, strategy="hamming", stream_end=end_time
        )
        audio_aggregation = blocks.DelayedAggregation(
            self.config.step, self.config.latency, strategy="first", stream_end=end_time
        )
        binarize = blocks.Binarize(source.uri, self.config.tau_active)
        return [
            # Identify global speakers with online clustering
            ops.starmap(lambda wav, seg, emb: (wav, clustering(seg, emb))),
            # Buffer 'num_overlapping' sliding chunks with a step of 1 chunk
            dops.buffer_slide(pred_aggregation.num_overlapping_windows),
            # Aggregate overlapping output windows
            ops.map(utils.unzip),
            ops.starmap(lambda wav_buffer, pred_buffer: (
                audio_aggregation(wav_buffer), pred_aggregation(pred_buffer)
            )),
            # Binarize output
            ops.starmap(lambda wav, pred: (binarize(pred), wav)),
        ]


class OnlineSpeakerDiarization:
    def __init__(self, config: Optional[PipelineConfig] = None, profile: bool = False):
        self.config = PipelineConfig() if config is None else config
        self.profile = profile
        self.segmentation = blocks.SpeakerSegmentation(config.segmentation, config.device)
        self.embedding = blocks.OverlapAwareSpeakerEmbedding(
            config.embedding, config.gamma, config.beta, norm=1, device=config.device
        )
        self.speaker_tracking = OnlineSpeakerTracking(config)
        msg = f"Latency should be in the range [{config.step}, {config.duration}]"
        assert config.step <= config.latency <= config.duration, msg

    def from_audio_source(self, source: src.AudioSource) -> rx.Observable:
        msg = f"Audio source has sample rate {source.sample_rate}, expected {self.config.sample_rate}"
        assert source.sample_rate == self.config.sample_rate, msg
        operators = []
        # Regularize the stream to a specific chunk duration and step
        if not source.is_regular:
            operators.append(dops.regularize_audio_stream(
                self.config.duration, self.config.step, source.sample_rate
            ))
        operators += [
            # Extract segmentation and keep audio
            ops.map(lambda wav: (wav, self.segmentation(wav))),
            # Extract embeddings and keep segmentation
            ops.starmap(lambda wav, seg: (wav, seg, self.embedding(wav, seg))),
        ]
        # Add speaker tracking
        operators += self.speaker_tracking.get_operators(source)
        if self.profile:
            return dops.profile(source.stream, operators)
        return source.stream.pipe(*operators)

    def from_feature_source(self, source: src.AudioSource) -> rx.Observable:
        return source.stream.pipe(*self.speaker_tracking.get_operators(source))
