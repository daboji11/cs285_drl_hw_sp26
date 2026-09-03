from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


exp_dir = Path("exp")


def find_run(prefix):
    matches = list(exp_dir.glob(f"{prefix}*"))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one match for {prefix}, got {matches}"
        )

    return matches[0] / "log.csv"


def plot_halfcheetah():
    no_baseline_path = find_run(
        "HalfCheetah-v4_cheetah_sd1_"
    )

    baseline_path = find_run(
        "HalfCheetah-v4_cheetah_baseline_sd1_"
    )

    no_baseline_df = pd.read_csv(no_baseline_path)
    baseline_df = pd.read_csv(baseline_path)

    # -------------------------
    # Plot 1: Eval Return
    # -------------------------

    plt.figure(figsize=(8, 7))

    plt.plot(
        no_baseline_df["Train_EnvstepsSoFar"],
        no_baseline_df["Eval_AverageReturn"],
        label="No Baseline",
    )

    plt.plot(
        baseline_df["Train_EnvstepsSoFar"],
        baseline_df["Eval_AverageReturn"],
        label="Baseline",
    )

    plt.xlabel("Environment Steps")
    plt.ylabel("Eval Average Return")
    plt.title("HalfCheetah - Eval Return")
    plt.legend()
    plt.tight_layout()

    plt.savefig("halfcheetah_eval_return.png")
    plt.show()


    # -------------------------
    # Plot 2: Baseline Loss
    # -------------------------

    plt.figure(figsize=(8, 7))

    plt.plot(
        baseline_df["Train_EnvstepsSoFar"],
        baseline_df["Baseline Loss"],
    )

    plt.xlabel("Environment Steps")
    plt.ylabel("Baseline Loss")
    plt.title("HalfCheetah - Baseline Loss")
    plt.tight_layout()

    plt.savefig("halfcheetah_baseline_loss.png")
    plt.show()

plot_halfcheetah()