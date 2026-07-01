import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = r"D:\Python Projects\A_Final_Final_Final\EXP_siglip2_Att_GRU_Ex"

OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
RESULTS_DIR = os.path.join(PROJECT_DIR, "image_results")


NAME_MAP = {
    "EXP_siglip2_Att_GRU_Ex": "SigLIP2 + Att-GRU_Ex",
}

MODEL_FOLDERS = {}
output_dir = os.path.join(PROJECT_DIR, "output")
if os.path.isdir(output_dir):
    MODEL_FOLDERS[
        NAME_MAP.get(os.path.basename(PROJECT_DIR), os.path.basename(PROJECT_DIR))
    ] = output_dir


def get_latest_file(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_training_data():
    data = {}
    for name, folder in MODEL_FOLDERS.items():
        latest = get_latest_file(folder, "training_log_*.xlsx")
        if not latest:
            latest = get_latest_file(folder, "training_log_*.csv")
        if latest:
            df = (
                pd.read_excel(latest)
                if latest.endswith(".xlsx")
                else pd.read_csv(latest)
            )
            df = df[df["Epoch"] <= 20].copy()
            data[name] = df
            print(
                f"Loaded {name}: {latest} ({df['Epoch'].min()}-{df['Epoch'].max()} epochs)"
            )
        else:
            print(f"No training log found for {name}")
    return data


def load_test_metrics():
    data = {}
    for name, folder in MODEL_FOLDERS.items():
        test_file = os.path.join(folder, "test_overall_averages.csv")
        if os.path.exists(test_file):
            df = pd.read_csv(test_file)
            data[name] = df.iloc[0].to_dict()
            print(f"Loaded test metrics for {name}")
        else:
            print(f"No test results for {name}")
    return data


def plot_training_curves(training_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10.colors

    has_val_loss = any("Val Loss" in df.columns for df in training_data.values())
    has_bleu1 = any("BLEU-1" in df.columns for df in training_data.values())

    for idx, (name, df) in enumerate(training_data.items()):
        color = colors[idx % len(colors)]
        epochs = df["Epoch"]
        axes[0].plot(
            epochs, df["Train Loss"], "-o", label=name, color=color, markersize=4
        )

        if has_val_loss:
            axes[1].plot(
                epochs, df["Val Loss"], "-o", label=name, color=color, markersize=4
            )
            axes[1].set_ylabel("Validation Loss")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlim(1, max(df["Epoch"].max() for df in training_data.values()))
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Validation Metric")
    axes[1].set_xlim(1, max(df["Epoch"].max() for df in training_data.values()))
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(
        os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches="tight"
    )
    print("Saved training_curves.png")
    plt.close()


def plot_bleu1_curve(training_data):
    has_bleu1 = any("BLEU-1" in df.columns for df in training_data.values())
    if not has_bleu1:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10.colors

    for idx, (name, df) in enumerate(training_data.items()):
        if "BLEU-1" not in df.columns:
            continue
        color = colors[idx % len(colors)]
        ax.plot(df["Epoch"], df["BLEU-1"], "-o", label=name, color=color, markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("BLEU-1")
    ax.set_title("BLEU-1 Score Over Training")
    ax.legend(fontsize=10)
    ax.grid(True)

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(
        os.path.join(RESULTS_DIR, "bleu1_curve.png"), dpi=150, bbox_inches="tight"
    )
    print("Saved bleu1_curve.png")
    plt.close()


def plot_metrics_bar(test_metrics):
    if not test_metrics:
        print("No test metrics to plot")
        return

    models = list(test_metrics.keys())
    skip_keys = {"samples", "img_name", "gen_caption"}
    metrics_names = [k for k in test_metrics[models[0]] if k not in skip_keys]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(models))
    width = 0.2

    for i, metric in enumerate(metrics_names):
        values = [test_metrics[m].get(metric, 0) for m in models]
        bars = ax.bar([xi + i * width for xi in x], values, width, label=metric.upper())
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Test Metrics")
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")

    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(
        os.path.join(RESULTS_DIR, "metrics_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved metrics_comparison.png")
    plt.close()


def plot_metrics_table(test_metrics):
    if not test_metrics:
        return

    df = pd.DataFrame(test_metrics).T
    bleu_cols = [c for c in df.columns if c.startswith("bleu")]
    other_cols = [c for c in df.columns if not c.startswith("bleu") and c != "samples"]
    display_cols = sorted(bleu_cols) + sorted(other_cols)
    display_cols = [c for c in display_cols if c in df.columns]
    df = df[display_cols]
    df.index.name = "Model"
    df = df.reset_index()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=[
            [f"{v:.4f}" if isinstance(v, float) else v for v in row]
            for row in df.values
        ],
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.3, 1.8)

    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", weight="bold")

    for j in range(1, len(df.columns)):
        col_vals = df.iloc[:, j]
        if col_vals.dtype in ["float64", "int64"]:
            max_idx = col_vals.idxmax() + 1
            table[(max_idx, j)].set_text_props(weight="bold")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(
        os.path.join(RESULTS_DIR, "metrics_table.png"), dpi=150, bbox_inches="tight"
    )
    print("Saved metrics_table.png")

    print("\n" + "=" * 70)
    print("TEST METRICS SUMMARY")
    print("=" * 70)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading training data...")
    training_data = load_training_data()

    print("\nLoading test metrics...")
    test_metrics = load_test_metrics()

    print("\nGenerating plots...")
    common_models = (
        set(training_data) & set(test_metrics) if test_metrics else set(training_data)
    )
    filtered_training = {k: v for k, v in training_data.items() if k in common_models}

    if filtered_training:
        plot_training_curves(filtered_training)
        plot_bleu1_curve(filtered_training)

    if test_metrics:
        plot_metrics_bar(test_metrics)
        plot_metrics_table(test_metrics)

    print(f"\nAll results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
