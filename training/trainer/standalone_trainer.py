"""Single-process training entry point.

Scaffold placeholder. Migration note: port from F1tenth-Genesis
`standalone_trainer.py`.

Run:
    python -m training.trainer.standalone_trainer --num-envs 512 --total-steps 500000
"""

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="F1TENTH RL trainer (placeholder).")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--config", type=str, default="training/configs/default.yaml")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    # TODO: build config, env (f1tenth_env), algorithm (qrsac), and run the loop.
    # Checkpoints are written under outputs/ (gitignored) and later loaded by the
    # on-car racing_rl node.
    raise SystemExit(
        "Scaffold placeholder: training loop not implemented yet "
        f"(num_envs={args.num_envs}, total_steps={args.total_steps})."
    )


if __name__ == "__main__":
    main()
