import argparse
from pathlib import Path

import optuna
import torch
from optuna.samplers import TPESampler

import diart.argdoc as argdoc
from diart.optim import Optimizer, HyperParameter
from diart.pipelines import PipelineConfig


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=str, help="Directory with audio files CONVERSATION.(wav|flac|m4a|...)")
    parser.add_argument("--reference", required=True, type=str,
                        help="Directory with RTTM files CONVERSATION.rttm. Names must match audio files")
    parser.add_argument("--step", default=0.5, type=float, help=f"{argdoc.STEP}. Defaults to 0.5")
    parser.add_argument("--latency", default=0.5, type=float, help=f"{argdoc.LATENCY}. Defaults to 0.5")
    parser.add_argument("--tau", default=0.5, type=float, help=f"{argdoc.TAU}. Defaults to 0.5")
    parser.add_argument("--rho", default=0.3, type=float, help=f"{argdoc.RHO}. Defaults to 0.3")
    parser.add_argument("--delta", default=1, type=float, help=f"{argdoc.DELTA}. Defaults to 1")
    parser.add_argument("--gamma", default=3, type=float, help=f"{argdoc.GAMMA}. Defaults to 3")
    parser.add_argument("--beta", default=10, type=float, help=f"{argdoc.BETA}. Defaults to 10")
    parser.add_argument("--max-speakers", default=20, type=int, help=f"{argdoc.MAX_SPEAKERS}. Defaults to 20")
    parser.add_argument("--batch-size", default=32, type=int, help=f"{argdoc.BATCH_SIZE}. Defaults to 32")
    parser.add_argument("--cpu", dest="cpu", action="store_true",
                        help=f"{argdoc.CPU}. Defaults to GPU if available, CPU otherwise")
    parser.add_argument("--hparams", nargs="+", default=("tau_active", "rho_update", "delta_new"),
                        help="Hyper-parameters to optimize. Must match names in `PipelineConfig`. Defaults to tau_active, rho_update and delta_new")
    parser.add_argument("--num-iter", default=100, type=int, help="Number of optimization trials")
    parser.add_argument("--storage", type=str,
                        help="Optuna storage string. If provided, continue a previous study instead of creating one. The database name must match the study name")
    parser.add_argument("--output", type=str, help="Working directory")
    args = parser.parse_args()
    args.device = torch.device("cpu") if args.cpu else None

    # Create the base configuration for each trial
    base_config = PipelineConfig.from_namespace(args)

    # Create hyper-parameters to optimize
    hparams = [HyperParameter.from_name(name) for name in args.hparams]

    # Use a custom storage if given
    if args.output is not None:
        msg = "Both `output` and `storage` were set, but only one was expected"
        assert args.storage is None, msg
        args.output = Path(args.output)
        args.output.mkdir(parents=True, exist_ok=True)
        study_or_path = args.output
    elif args.storage is not None:
        db_name = Path(args.storage).stem
        study_or_path = optuna.load_study(db_name, args.storage, TPESampler())
    else:
        msg = "Please provide either `output` or `storage`"
        raise ValueError(msg)

    # Run optimization
    Optimizer(
        speech_path=args.root,
        reference_path=args.reference,
        study_or_path=study_or_path,
        batch_size=args.batch_size,
        hparams=hparams,
        base_config=base_config,
    )(num_iter=args.num_iter, show_progress=True)


if __name__ == "__main__":
    run()
