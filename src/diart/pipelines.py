from pathlib import Path
from typing import Optional, Union, Text

import numpy as np
import rx
import rx.operators as ops
import torch
from einops import rearrange
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import SlidingWindowFeature, SlidingWindow

from . import functional as fn
from . import operators as dops
from . import sources as src


class PipelineConfig:
    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        embedding: PipelineModel = "pyannote/embedding",
        duration: Optional[float] = None,
        step: float = 0.5,
        latency: Optional[float] = None,
        tau_active: float = 0.6,
        rho_update: float = 0.3,
        delta_new: float = 1,
        gamma: float = 3,
        beta: float = 10,
        max_speakers: int = 20,
    ):
        self.segmentation = fn.FrameWiseModel(segmentation)
        self.duration = duration
        if self.duration is None:
            self.duration = self.segmentation.model.specifications.duration
        self.embedding = fn.ChunkWiseModel(embedding)
        self.step = step
        self.latency = latency
        if self.latency is None:
            self.latency = self.step
        assert self.step <= self.latency <= self.duration, "Invalid latency requested"
        self.tau_active = tau_active
        self.rho_update = rho_update
        self.delta_new = delta_new
        self.gamma = gamma
        self.beta = beta
        self.max_speakers = max_speakers

    @property
    def sample_rate(self) -> int:
        return self.segmentation.model.audio.sample_rate

    def get_end_time(self, source: src.AudioSource) -> Optional[float]:
        if source.duration is not None:
            return source.duration - source.duration % self.step
        return None


class OnlineSpeakerTracking:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def from_model_streams(
        self,
        uri: Text,
        end_time: Optional[float],
        segmentation_stream: rx.Observable,
        embedding_stream: rx.Observable,
        audio_chunk_stream: Optional[rx.Observable] = None,
    ) -> rx.Observable:
        # Initialize clustering and aggregation modules
        clustering = fn.OnlineSpeakerClustering(
            self.config.tau_active,
            self.config.rho_update,
            self.config.delta_new,
            "cosine",
            self.config.max_speakers,
        )
        aggregation = fn.DelayedAggregation(
            self.config.step, self.config.latency, strategy="hamming", stream_end=end_time
        )
        binarize = fn.Binarize(uri, self.config.tau_active)

        # Join segmentation and embedding streams to update a background clustering model
        #  while regulating latency and binarizing the output
        pipeline = rx.zip(segmentation_stream, embedding_stream).pipe(
            ops.starmap(clustering),
            # Buffer 'num_overlapping' sliding chunks with a step of 1 chunk
            dops.buffer_slide(aggregation.num_overlapping_windows),
            # Aggregate overlapping output windows
            ops.map(aggregation),
            # Binarize output
            ops.map(binarize),
        )
        # Add corresponding waveform to the output
        if audio_chunk_stream is not None:
            window_selector = fn.DelayedAggregation(
                self.config.step, self.config.latency, strategy="first", stream_end=end_time
            )
            waveform_stream = audio_chunk_stream.pipe(
                dops.buffer_slide(window_selector.num_overlapping_windows),
                ops.map(window_selector),
            )
            return rx.zip(pipeline, waveform_stream)
        # No waveform needed, add None for consistency
        return pipeline.pipe(ops.map(lambda ann: (ann, None)))


class OnlineSpeakerDiarization:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.speaker_tracking = OnlineSpeakerTracking(config)

    def from_source(
        self,
        source: src.AudioSource,
        output_waveform: bool = True
    ) -> rx.Observable:
        msg = f"Audio source has sample rate {source.sample_rate}, expected {self.config.sample_rate}"
        assert source.sample_rate == self.config.sample_rate, msg
        # Regularize the stream to a specific chunk duration and step
        regular_stream = source.stream
        if not source.is_regular:
            regular_stream = source.stream.pipe(
                dops.regularize_stream(self.config.duration, self.config.step, source.sample_rate)
            )
        # Branch the stream to calculate chunk segmentation
        segmentation_stream = regular_stream.pipe(ops.map(self.config.segmentation))
        # Join audio and segmentation stream to calculate speaker embeddings
        osp = fn.OverlappedSpeechPenalty(gamma=self.config.gamma, beta=self.config.beta)
        embedding_stream = rx.zip(regular_stream, segmentation_stream).pipe(
            ops.starmap(lambda wave, seg: (wave, osp(seg))),
            ops.starmap(self.config.embedding),
            ops.map(fn.EmbeddingNormalization(norm=1))
        )
        end_time = self.config.get_end_time(source)
        chunk_stream = regular_stream if output_waveform else None
        return self.speaker_tracking.from_model_streams(
            source.uri, end_time, segmentation_stream, embedding_stream, chunk_stream
        )

    def from_file(
        self,
        file: Union[Text, Path],
        output_waveform: bool = False,
        batch_size: int = 32
    ) -> rx.Observable:
        # Audio file information
        file = Path(file)
        chunk_loader = src.ChunkLoader(
            self.config.sample_rate, self.config.duration, self.config.step
        )
        uri = file.name.split(".")[0]
        end_time = chunk_loader.audio.get_duration(file) % self.config.step

        # Initialize pipeline modules
        osp = fn.OverlappedSpeechPenalty(self.config.gamma, self.config.beta)
        emb_norm = fn.EmbeddingNormalization(norm=1)

        # Pre-calculate segmentation and embeddings
        chunks = rearrange(
            chunk_loader.get_chunks(file),
            "chunk channel sample -> chunk sample channel"
        )
        num_chunks = chunks.shape[0]
        segmentation, embeddings = [], []
        for i in range(0, num_chunks, batch_size):
            i_end = i + batch_size
            if i_end > num_chunks:
                i_end = num_chunks
            batch = chunks[i:i_end]
            seg = self.config.segmentation(batch)
            segmentation.append(seg)
            embeddings.append(emb_norm(self.config.embedding(batch, osp(seg))))
        segmentation = np.vstack(segmentation)
        embeddings = torch.vstack(embeddings)

        # Stream pre-calculated segmentation, embeddings and chunks
        resolution = self.config.duration / segmentation.shape[1]
        segmentation_stream = rx.range(0, num_chunks).pipe(
            ops.map(lambda i: SlidingWindowFeature(
                segmentation[i], SlidingWindow(resolution, resolution, i * self.config.step)
            ))
        )
        embedding_stream = rx.range(0, num_chunks).pipe(ops.map(lambda i: embeddings[i]))
        wav_resolution = 1 / self.config.sample_rate
        chunk_stream = None
        if output_waveform:
            chunk_stream = rx.range(0, num_chunks).pipe(
                ops.map(lambda i: SlidingWindowFeature(
                    chunks[i], SlidingWindow(wav_resolution, wav_resolution, i * self.config.step)
                ))
            )

        return self.speaker_tracking.from_model_streams(
            uri, end_time, segmentation_stream, embedding_stream, chunk_stream
        )
