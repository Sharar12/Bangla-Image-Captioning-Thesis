import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = r"D:\Python Projects\Research Papers Test"

THESIS_DIR = r"D:\Python Projects\Train Full dataset with final model - For Thesis"

EXCLUDE_DIRS = {"ResearchP", "siglip2_banglatdm", "siglip2_Att_GRU"}

NAME_MAP = {
    "siglip2_banglabert": "SigLIP2 + BanglaBERT",
    "siglip2_banglagpt": "SigLIP2 + BanglaGPT",
    "siglip2_banglatdm": "SigLIP2 + BanglaTDM",
    "Siglip2_cusDecoder": "SigLIP2 + Custom Decoder",
    "siglip2_Att_GRU_Ex": "SigLIP2 + Att-GRU_Ex",
}

MODEL_FOLDERS = {}
for search_dir in [BASE_DIR, THESIS_DIR]:
    if not os.path.exists(search_dir):
        continue
    for entry in os.listdir(search_dir):
        entry_path = os.path.join(search_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry in EXCLUDE_DIRS:
            continue
        name = NAME_MAP.get(entry, entry.replace("_", " ").title())
        if name in MODEL_FOLDERS:
            continue
        output_dir = os.path.join(entry_path, "output")
        if os.path.isdir(output_dir):
            MODEL_FOLDERS[name] = output_dir


def get_latest_file(folder, pattern):
    files = glob.glob(os.path.join(folder, pattern))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_training_data():
    data = {}
    for name, folder in MODEL_FOLDERS.items():
        latest = get_latest_file(folder, "training_log_*.xlsx")
        if latest:
            df = pd.read_excel(latest)
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

    for idx, (name, df) in enumerate(training_data.items()):
        color = colors[idx % len(colors)]
        epochs = df["Epoch"]

        axes[0].plot(
            epochs, df["Train Loss"], "-o", label=name, color=color, markersize=4
        )
        axes[1].plot(
            epochs, df["Val Loss"], "-o", label=name, color=color, markersize=4
        )

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Loss")
    axes[0].set_title("Training Loss Comparison (20 Epochs)")
    axes[0].set_xlim(1, 20)
    axes[0].legend(fontsize=8)
    axes[0].grid(True)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Loss")
    axes[1].set_title("Validation Loss Comparison (20 Epochs)")
    axes[1].set_xlim(1, 20)
    axes[1].legend(fontsize=8)
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(BASE_DIR, "image_results", "training_curves.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved training_curves.png")
    plt.close()


def plot_metrics_bar(test_metrics):
    if not test_metrics:
        print("No test metrics to plot")
        return

    models = list(test_metrics.keys())
    metrics_names = ["bleu1", "meteor", "rouge_l", "cider"]

    fig, ax = plt.subplots(figsize=(14, 6))

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
                fontsize=7,
            )

    ax.set_xlabel("Model (Vision Encoder + Text Decoder)")
    ax.set_ylabel("Score")
    ax.set_title("Test Metrics Comparison (20 Epochs)")
    ax.set_xticks([xi + 1.5 * width for xi in x])
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(BASE_DIR, "image_results", "metrics_comparison.png"),
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

    fig, ax = plt.subplots(figsize=(14, 6))
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
    table.set_fontsize(8)
    table.scale(1.3, 1.8)

    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", weight="bold")

    for j in range(1, len(df.columns)):
        col_vals = df.iloc[:, j]
        if col_vals.dtype in ["float64", "int64"]:
            max_idx = col_vals.idxmax() + 1
            table[(max_idx, j)].set_text_props(weight="bold")

    plt.savefig(
        os.path.join(BASE_DIR, "image_results", "metrics_table.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved metrics_table.png")

    print("\n" + "=" * 70)
    print("TEST METRICS SUMMARY (20 Epochs)")
    print("=" * 70)
    pd.set_option("display.float_format", "{:.4f}".format)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")


def plot_training_time(training_data):
    if not training_data:
        return

    total_times = {}
    last_epochs = {}
    for name, df in training_data.items():
        last_epochs[name] = df["Epoch"].max()
        if "Time (s)" in df.columns:
            total_times[name] = df["Time (s)"].sum()
        elif "Time" in df.columns:
            total_times[name] = df["Time"].sum()
        elif "Elapsed Time" in df.columns:
            total_times[name] = df["Elapsed Time"].sum()
        elif "time" in df.columns:
            total_times[name] = df["time"].sum()
        else:
            print(
                f"No time column found for {name}, available columns: {df.columns.tolist()}"
            )
            continue

    if not total_times:
        print("No time data found")
        return

    models = list(total_times.keys())
    times = list(total_times.values())

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10.colors[: len(models)]
    bars = ax.bar(models, times, color=colors)

    for bar, t, m in zip(bars, times, models):
        hours = int(t // 3600)
        minutes = int((t % 3600) // 60)
        seconds = int(t % 60)
        label = (
            f"{hours}h {minutes}m {seconds}s\n({int(last_epochs[m])} epochs)"
            if hours > 0
            else f"{minutes}m {seconds}s\n({int(last_epochs[m])} epochs)"
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.01,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel("Model (Vision Encoder + Text Decoder)")
    ax.set_ylabel("Total Training Time (seconds)")
    ax.set_title("Total Training Time per Model (Up to 20 Epochs)")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, max(times) + 1000)
    ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(BASE_DIR, "image_results", "training_time.png"),
        dpi=150,
        bbox_inches="tight",
    )
    print("Saved training_time.png")
    plt.close()


def main():
    output_dir = os.path.join(BASE_DIR, "image_results")
    os.makedirs(output_dir, exist_ok=True)

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
        plot_training_time(filtered_training)

    if test_metrics:
        plot_metrics_bar(test_metrics)
        plot_metrics_table(test_metrics)

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
