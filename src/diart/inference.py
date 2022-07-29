from pathlib import Path
from typing import Union, Text, Optional, Callable, Tuple

import pandas as pd
import rx.operators as ops
from pyannote.core import Annotation, SlidingWindowFeature
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from rx.core import Observer

import diart.operators as dops
import diart.sources as src
from diart.pipelines import OnlineSpeakerDiarization
from diart.sinks import RTTMAccumulator, RTTMWriter, RealTimePlot


class RealTimeInference:
    """
    Simplifies inference in real time for users that do not want to play with the reactivex interface.
    Streams an audio source to an online speaker diarization pipeline.
    It allows users to attach a chain of operations in the form of hooks.

    Parameters
    ----------
    pipeline: OnlineSpeakerDiarization
        Configured speaker diarization pipeline.
    source: AudioSource
        Audio source to be read and streamed.
    do_plot: bool
        Whether to draw predictions in a moving plot. Defaults to True.
    """
    def __init__(
        self,
        pipeline: OnlineSpeakerDiarization,
        source: src.AudioSource,
        do_plot: bool = True
    ):
        self.pipeline = pipeline
        self.source = source
        self.do_plot = do_plot
        self.accumulator = RTTMAccumulator()
        self.stream = self.pipeline.from_audio_source(source).pipe(
            dops.progress(f"Streaming {source.uri}", total=source.length, leave=True),
            ops.do(self.accumulator),
        )

    def attach_hooks(self, *hooks: Callable[[Tuple[Annotation, SlidingWindowFeature]], None]):
        """
        Attach hooks to the pipeline.

        Parameters
        ----------
        *hooks: (Tuple[Annotation, SlidingWindowFeature]) -> None
            Hook functions to consume emitted annotations and audio.
        """
        self.stream = self.stream.pipe(*[ops.do_action(hook) for hook in hooks])

    def attach_observers(self, *observers: Observer):
        """
        Attach rx observers to the pipeline.

        Parameters
        ----------
        *observers: Observer
            Observers to consume emitted annotations and audio.
        """
        self.stream = self.stream.pipe(*[ops.do(sink) for sink in observers])

    def __call__(self) -> Annotation:
        """
        Stream audio chunks from `source` to `pipeline`
        writing predictions to disk.

        Returns
        -------
        predictions: Annotation
            Speaker diarization pipeline predictions
        """
        config = self.pipeline.config
        observable = self.stream
        if self.do_plot:
            # Buffering is needed for the real-time plot, so we do this at the very end
            observable = self.stream.pipe(
                dops.buffer_output(
                    duration=config.duration,
                    step=config.step,
                    latency=config.latency,
                    sample_rate=config.sample_rate,
                ),
                ops.do(RealTimePlot(config.duration, config.latency)),
            )
        observable.subscribe()
        self.source.read()
        return self.accumulator.annotation


class Benchmark:
    """
    Run an online speaker diarization pipeline on a set of audio files in batches.
    It writes predictions to a given output directory.

    If the reference is given, it calculates the average diarization error rate.

    Parameters
    ----------
    speech_path: Text or Path
        Directory with audio files.
    reference_path: Text, Path or None
        Directory with reference RTTM files (same names as audio files).
        If None, performance will not be calculated.
        Defaults to None.
    output_path: Text, Path or None
        Output directory to store predictions in RTTM format.
        If None, predictions will not be written to disk.
        Defaults to None.
    show_progress: bool
        Whether to show progress bars.
        Defaults to True.
    show_report: bool
        Whether to print a performance report to stdout.
        Defaults to True.
    batch_size: int
        Inference batch size.
        If < 2, then it will run in real time.
        If >= 2, then it will pre-calculate segmentation and
        embeddings, running the rest in real time.
        The performance between this two modes does not differ.
        Defaults to 32.
    """
    def __init__(
        self,
        speech_path: Union[Text, Path],
        reference_path: Optional[Union[Text, Path]] = None,
        output_path: Optional[Union[Text, Path]] = None,
        show_progress: bool = True,
        show_report: bool = True,
        batch_size: int = 32,
    ):
        self.speech_path = Path(speech_path).expanduser()
        assert self.speech_path.is_dir(), "Speech path must be a directory"

        self.reference_path = reference_path
        if reference_path is not None:
            self.reference_path = Path(self.reference_path).expanduser()
            assert self.reference_path.is_dir(), "Reference path must be a directory"

        self.output_path = output_path
        if self.output_path is not None:
            self.output_path = Path(output_path).expanduser()
            self.output_path.mkdir(parents=True, exist_ok=True)

        self.show_progress = show_progress
        self.show_report = show_report
        self.batch_size = batch_size

    def __call__(self, pipeline: OnlineSpeakerDiarization) -> Optional[pd.DataFrame]:
        """
        Run a given pipeline on a set of audio files using
        pre-calculated segmentation and embeddings in batches.

        Parameters
        ----------
        pipeline: OnlineSpeakerDiarization
            Configured speaker diarization pipeline.

        Returns
        -------
        performance: pandas.DataFrame, optional
            DataFrame with detailed performance on each file, as well as average performance.
            None if the reference is not provided.
        """
        loader = src.AudioLoader(pipeline.config.sample_rate, mono=True)
        audio_file_paths = list(self.speech_path.iterdir())
        num_audio_files = len(audio_file_paths)
        predictions = []
        for i, filepath in enumerate(audio_file_paths):
            num_chunks = loader.get_num_sliding_chunks(
                filepath, pipeline.config.duration, pipeline.config.step
            )

            # Stream fully online if batch size is 1 or lower
            if self.batch_size < 2:
                source = src.FileAudioSource(
                    filepath,
                    pipeline.config.sample_rate,
                    pipeline.config.duration,
                    pipeline.config.step,
                )
                observable = pipeline.from_audio_source(source)
            else:
                msg = f"Pre-calculating {filepath.stem} ({i + 1}/{num_audio_files})"
                source = src.PrecalculatedFeaturesAudioSource(
                    filepath,
                    pipeline.config.sample_rate,
                    pipeline.segmentation,
                    pipeline.embedding,
                    pipeline.config.duration,
                    pipeline.config.step,
                    self.batch_size,
                    progress_msg=msg if self.show_progress else None,
                )
                observable = pipeline.from_feature_source(source)

            if self.show_progress:
                observable = observable.pipe(
                    dops.progress(
                        desc=f"Streaming {source.uri} ({i + 1}/{num_audio_files})",
                        total=num_chunks,
                        leave=False,
                    )
                )

            if self.output_path is not None:
                observable = observable.pipe(
                    ops.do(RTTMWriter(self.output_path / f"{source.uri}.rttm"))
                )

            accumulator = RTTMAccumulator()
            observable.subscribe(accumulator)
            source.read()
            predictions.append(accumulator.annotation)

        # Run evaluation
        if self.reference_path is not None:
            metric = DiarizationErrorRate(collar=0, skip_overlap=False)
            for hyp in predictions:
                ref = load_rttm(self.reference_path / f"{hyp.uri}.rttm").popitem()[1]
                metric(ref, hyp)

            return metric.report(display=self.show_report)
