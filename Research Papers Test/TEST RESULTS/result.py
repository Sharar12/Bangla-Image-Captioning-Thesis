import os
import sys
import warnings
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR = Path(r"D:\Python Projects\Research Papers Test")
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "BNNATURE" / "images"

PROJECTS = [
    ("Siglip2_cusDecoder", "SigLIP2 + Custom Transformer Decoder", True),
    ("swin_banglabert", "Swin-Tiny + BanglaBERT", True),
    ("vit_banglabert", "ViT-Base + BanglaBERT", True),
    ("inception_gru", "InceptionV3 + GRU", True),
    ("xception_bigru_attention", "Xception + Bi-GRU + Attention", True),
    ("clip_banglagpt", "CLIP ViT-L/14 + BanglaGPT", False),
    ("ResearchP", "ViT-Base/384 + BanglaBERT (Enc-Dec)", True),
]

COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1"]
METRIC_COLORS = {
    "bleu1": "#4e79a7",
    "bleu2": "#f28e2b",
    "bleu3": "#e15759",
    "bleu4": "#76b7b2",
    "meteor": "#59a14f",
    "rouge_l": "#edc948",
    "cider": "#b07aa1",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def read_training_logs(project_dir, project_label):
    log_dir = project_dir / "output"
    if not log_dir.exists():
        return None
    xlsx_files = sorted(log_dir.glob("training_log_*.xlsx"))
    csv_files = sorted(log_dir.glob("training_log_*.csv"))
    candidates = xlsx_files + csv_files
    if not candidates:
        return None
    latest = candidates[-1]
    try:
        if latest.suffix == ".xlsx":
            df = pd.read_excel(latest)
        else:
            df = pd.read_csv(latest)
        required = {"Epoch", "Train Loss", "Val Loss"}
        if not required.issubset(df.columns):
            return None
        df = df.dropna(subset=["Epoch"])
        df["Project"] = project_label
        return df
    except Exception:
        return None


def read_test_results(project_dir):
    csv_path = project_dir / "output" / "test_results.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def read_detailed_results(project_dir):
    csv_path = project_dir / "output" / "detailed_results.csv"
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def collect_training_data():
    all_dfs = []
    for folder, label, _ in PROJECTS:
        df = read_training_logs(BASE_DIR / folder, label)
        if df is not None:
            all_dfs.append(df)
            print(f"  [OK] {label}")
        else:
            print(f"  [--] {label} (no training log)")
    if not all_dfs:
        return None
    return pd.concat(all_dfs, ignore_index=True)


def collect_test_data():
    results = []
    for folder, label, _ in PROJECTS:
        df = read_test_results(BASE_DIR / folder)
        if df is not None:
            results.append((label, df.iloc[0].to_dict(), folder))
            print(f"  [OK] {label}")
        else:
            print(f"  [--] {label} (no test results)")
    return results


def plot_training_curves(train_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title, ylabel in [
        (axes[0], ["Train Loss", "Val Loss"], "Training & Validation Loss", "Loss"),
        (axes[1], ["Learning Rate"], "Learning Rate Schedule", "Learning Rate"),
    ]:
        for i, proj in enumerate(train_df["Project"].unique()):
            pdf = train_df[train_df["Project"] == proj].sort_values("Epoch")
            for col in metric:
                if col not in pdf.columns:
                    continue
                style = "-" if "Loss" in col else "--"
                alpha = 0.7 if "Loss" in col else 1.0
                label_col = f"{proj} ({col})" if len(metric) > 1 else proj
                ax.plot(
                    pdf["Epoch"],
                    pdf[col],
                    color=COLORS[i % len(COLORS)],
                    linestyle=style,
                    alpha=alpha,
                    label=label_col,
                    linewidth=1.5,
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best", framealpha=0.7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "training_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def plot_metric_comparison(test_data):
    all_metrics = ["bleu1", "bleu2", "bleu3", "bleu4", "meteor", "rouge_l", "cider"]
    projects = [label for label, _, _ in test_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    common = [m for m in ["bleu1", "meteor", "rouge_l", "cider"]]
    x = np.arange(len(projects))
    w = 0.2

    ax = axes[0]
    for i, metric in enumerate(common):
        vals = []
        for _, scores, _ in test_data:
            vals.append(scores.get(metric, 0))
        bars = ax.bar(
            x + i * w,
            vals,
            w,
            label=metric.upper(),
            color=list(METRIC_COLORS.values())[i],
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=45,
            )
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(projects, fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Common Metrics Comparison (BLEU-1, METEOR, ROUGE-L, CIDEr)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    rp_projects = []
    rp_metrics = []
    for label, scores, _ in test_data:
        has_all = all(m in scores for m in all_metrics)
        if has_all:
            rp_projects.append(label)
            rp_metrics.append([scores.get(m, 0) for m in all_metrics])
    if rp_metrics:
        rp_data = np.array(rp_metrics)
        x2 = np.arange(len(all_metrics))
        for i in range(len(rp_projects)):
            ax.plot(
                x2,
                rp_data[i],
                "o-",
                color=COLORS[i % len(COLORS)],
                label=rp_projects[i],
                linewidth=1.5,
                markersize=6,
            )
        ax.set_xticks(x2)
        ax.set_xticklabels([m.upper() for m in all_metrics], fontsize=8)
        ax.set_ylabel("Score")
        ax.set_title("Full BLEU Breakdown (ResearchP Style)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No project with full BLEU-1..4 metrics",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
        )

    plt.tight_layout()
    path = OUTPUT_DIR / "metric_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def plot_radar_chart(test_data):
    labels_list = ["bleu1", "meteor", "rouge_l", "cider"]
    projects = [label for label, _, _ in test_data]
    num_vars = len(labels_list)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    for i, (label, scores, _) in enumerate(test_data):
        vals = [scores.get(m, 0) for m in labels_list]
        vals += vals[:1]
        ax.plot(
            angles,
            vals,
            "o-",
            color=COLORS[i % len(COLORS)],
            label=label,
            linewidth=1.5,
            markersize=4,
        )
        ax.fill(angles, vals, alpha=0.05, color=COLORS[i % len(COLORS)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.upper() for m in labels_list], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Model Performance Radar", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "radar_chart.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def plot_metric_heatmap(test_data):
    labels_list = ["bleu1", "meteor", "rouge_l", "cider"]
    data = []
    labels = []
    for label, scores, _ in test_data:
        labels.append(label)
        data.append([scores.get(m, 0) for m in labels_list])

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels_list)))
    ax.set_xticklabels([m.upper() for m in labels_list], fontsize=10)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(data)):
        for j in range(len(data[i])):
            ax.text(
                j,
                i,
                f"{data[i][j]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if data[i][j] > 0.5 else "black",
            )

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Metric Scores Heatmap", fontsize=13)
    plt.tight_layout()
    path = OUTPUT_DIR / "metric_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def generate_summary_table(test_data, train_df):
    lines = []
    lines.append("=" * 90)
    lines.append(
        f"{'Model':45s} {'BLEU-1':>8s} {'METEOR':>8s} {'ROUGE-L':>8s} {'CIDEr':>8s} {'Best Epoch':>11s}"
    )
    lines.append("-" * 90)

    for label, scores, _ in test_data:
        b1 = f"{scores.get('bleu1', 0):.4f}"
        me = f"{scores.get('meteor', 0):.4f}"
        rl = f"{scores.get('rouge_l', 0):.4f}"
        cd = f"{scores.get('cider', 0):.4f}"
        best_epoch = ""
        if train_df is not None:
            pdf = train_df[train_df["Project"] == label]
            if not pdf.empty and "Val Loss" in pdf.columns:
                best = pdf.loc[pdf["Val Loss"].idxmin()]
                best_epoch = f"E{int(best['Epoch'])}"
        lines.append(
            f"{label:45s} {b1:>8s} {me:>8s} {rl:>8s} {cd:>8s} {best_epoch:>11s}"
        )

    lines.append("=" * 90)
    return "\n".join(lines)


def plot_sample_images(test_data, num_samples=3):
    image_dir = Path(r"D:\Python Projects\Research Papers Test\BNNATURE\images")
    fig, axes = plt.subplots(
        num_samples, len(test_data), figsize=(4 * len(test_data), 4 * num_samples)
    )
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if len(test_data) == 1:
        axes = axes.reshape(-1, 1)

    for col, (label, _, folder) in enumerate(test_data):
        detailed = read_detailed_results(BASE_DIR / folder)
        if detailed is None or detailed.empty:
            for row in range(num_samples):
                axes[row, col].text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=axes[row, col].transAxes,
                )
                axes[row, col].set_title(f"{label}", fontsize=9)
            continue

        samples = detailed.sample(min(num_samples, len(detailed)))
        for row, (_, sample) in enumerate(samples.iterrows()):
            ax = axes[row, col]
            img_name = sample.get("img_name", "")
            img_path = image_dir / img_name
            if img_path.exists():
                img = mpimg.imread(img_path)
                ax.imshow(img)
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No image\n{img_name}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=6,
                )
            hyp = str(sample.get("hypothesis", ""))
            ref = str(sample.get("reference", ""))
            if ref.startswith("[") and ref.endswith("]"):
                try:
                    ref_list = eval(ref)
                    ref = ref_list[0] if ref_list else ""
                except Exception:
                    pass
            ax.set_title(f"{label}", fontsize=8, fontweight="bold")
            ax.text(
                0.5,
                -0.15,
                f"Pred: {hyp[:60]}",
                transform=ax.transAxes,
                ha="center",
                fontsize=6,
                wrap=True,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f5e9"),
            )
            ax.text(
                0.5,
                -0.30,
                f"Ref:  {ref[:60]}",
                transform=ax.transAxes,
                ha="center",
                fontsize=6,
                wrap=True,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#e3f2fd"),
            )
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    path = OUTPUT_DIR / "sample_predictions.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def plot_loss_distribution(test_data):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    metrics_list = ["bleu1", "meteor", "rouge_l", "cider"]
    colors_plot = ["#4e79a7", "#59a14f", "#edc948", "#b07aa1"]

    for idx, metric in enumerate(metrics_list):
        ax = axes[idx]
        all_vals = []
        all_labels = []
        for label, _, folder in test_data:
            detailed = read_detailed_results(BASE_DIR / folder)
            if detailed is not None and metric in detailed.columns:
                vals = detailed[metric].dropna().values
                all_vals.append(vals)
                all_labels.append(label)

        if all_vals:
            parts = ax.violinplot(
                all_vals,
                positions=range(len(all_labels)),
                showmeans=True,
                showmedians=True,
            )
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors_plot[idx % len(colors_plot)])
                pc.set_alpha(0.6)
            ax.set_xticks(range(len(all_labels)))
            ax.set_xticklabels(all_labels, fontsize=7, rotation=20, ha="right")
            ax.set_ylabel(metric.upper())
            ax.set_title(f"{metric.upper()} Score Distribution")
            ax.grid(True, alpha=0.3, axis="y")
        else:
            ax.text(
                0.5,
                0.5,
                f"No {metric.upper()} data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    plt.tight_layout()
    path = OUTPUT_DIR / "score_distributions.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  BENGALI IMAGE CAPTIONING - RESULT ANALYSIS")
    print("=" * 60)

    print("\n[1/7] Collecting training logs...")
    train_df = collect_training_data()

    print("\n[2/7] Collecting test results...")
    test_data = collect_test_data()

    if not test_data:
        print("\nNo test results found. Run test.py for at least one project first.")
        sys.exit(1)

    print("\n[3/7] Plotting training curves...")
    if train_df is not None:
        plot_training_curves(train_df)
    else:
        print("  (no training logs available)")

    print("\n[4/7] Plotting metric comparisons...")
    plot_metric_comparison(test_data)
    plot_radar_chart(test_data)
    plot_metric_heatmap(test_data)

    print("\n[5/7] Plotting score distributions...")
    plot_loss_distribution(test_data)

    print("\n[6/7] Plotting sample predictions...")
    plot_sample_images(test_data, num_samples=3)

    print("\n[7/7] Generating summary table...")
    summary = generate_summary_table(test_data, train_df)
    print()
    print(summary)

    table_path = OUTPUT_DIR / "summary_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\n  Saved: {table_path.name}")

    print("\n" + "=" * 60)
    print("  ALL DONE! Check the 'TEST RESULTS' folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
