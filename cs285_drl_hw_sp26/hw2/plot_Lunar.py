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


def plot_lunar_lander_gae():
    log_path = find_run(
        "LunarLander-v2_lunar_lander_lambda0.99_sd1_20260903_162217"
    )

    df = pd.read_csv(log_path)

    plt.figure(figsize=(8, 7))

    plt.plot(
        df["Train_EnvstepsSoFar"],
        df["Eval_AverageReturn"],
        label="GAE λ = 0.99",
    )

    plt.xlabel("Environment Steps")
    plt.ylabel("Eval Average Return")
    plt.title("LunarLander - GAE")
    plt.legend()
    plt.tight_layout()

    plt.savefig("lunar_lander_gae_0.99.png")
    plt.show()


plot_lunar_lander_gae()