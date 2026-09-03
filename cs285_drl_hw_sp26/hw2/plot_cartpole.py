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


# -------------------------
# small batch
# -------------------------

small_runs = {
    "vanilla": "CartPole-v0_cartpole_sd1_",
    "rtg": "CartPole-v0_cartpole_rtg_sd1_",
    "na": "CartPole-v0_cartpole_na_sd1_",
    "rtg + na": "CartPole-v0_cartpole_rtg_na_sd1_",
}

plt.figure(figsize=(8, 10))

for label, prefix in small_runs.items():
    df = pd.read_csv(find_run(prefix))

    plt.plot(
        df["Train_EnvstepsSoFar"],
        df["Eval_AverageReturn"],
        label=label,
    )

plt.xlabel("Environment Steps")
plt.ylabel("Eval Average Return")
plt.title("CartPole - Small Batch")

plt.ylim(0, 220)

plt.legend()
plt.tight_layout()
plt.show()


# -------------------------
# large batch
# -------------------------

large_runs = {
    "vanilla": "CartPole-v0_cartpole_lb_sd1_",
    "rtg": "CartPole-v0_cartpole_lb_rtg_sd1_",
    "na": "CartPole-v0_cartpole_lb_na_sd1_",
    "rtg + na": "CartPole-v0_cartpole_lb_rtg_na_sd1_",
}

plt.figure(figsize=(8, 10))

for label, prefix in large_runs.items():
    df = pd.read_csv(find_run(prefix))

    plt.plot(
        df["Train_EnvstepsSoFar"],
        df["Eval_AverageReturn"],
        label=label,
    )

plt.xlabel("Environment Steps")
plt.ylabel("Eval Average Return")
plt.title("CartPole - Large Batch")

plt.ylim(0, 220)

plt.legend()
plt.tight_layout()
plt.show()