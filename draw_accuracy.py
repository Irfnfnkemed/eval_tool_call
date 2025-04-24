import argparse
import json
import pandas as pd
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

models = [
    "Llama-3.2-1B-Instruct-q0f16-MLC",
    "Llama-3.2-3B-Instruct-q0f16-MLC",
    "Llama-3.1-8B-Instruct-q0f16-MLC",
    "Llama-3.1-70B-Instruct-q0f16-MLC",
    "Qwen2.5-72B-Instruct-q0f16-MLC",
]
datasets = [
    "BFCL_v3_parallel",
    "BFCL_v3_live_multiple",
]


def draw(args: argparse.ArgumentParser, summary: Dict):
    colors = ["#0056b3", "#FF8C00"]

    # First figure - Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))  # Slightly wider figure
    bars = ["no_stag", "use_stag"]
    width = 0.35
    gap = 0.03
    x = np.arange(len(models))

    for ax_idx, ax in enumerate(axes.flat):
        dataset = datasets[ax_idx]
        draw_info = {}
        for model in summary:
            draw_info[model] = summary[model][dataset]
        for i, bar in enumerate(bars):
            values = [draw_info[model][bar]["CORRECT_CALL"] * 100 for model in models]
            ax.bar(x + i * (width + gap), values, width, color=colors[i], label=bar)
        ax.set_title(dataset.replace("_", "-"), fontsize=25, pad=20)  # Added pad
        ax.set_ylim(0, 105)
        ax.set_xticks(x + width / 2)
        ax.set_ylabel("Output Accuracy (%)", fontsize=18)
        ax.axhline(y=20, color="gray", linestyle="--", linewidth=1)
        ax.axhline(y=40, color="gray", linestyle="--", linewidth=1)
        ax.axhline(y=60, color="gray", linestyle="--", linewidth=1)
        ax.axhline(y=80, color="gray", linestyle="--", linewidth=1)
        ax.axhline(y=100, color="gray", linestyle="--", linewidth=2)
        ax.set_yticks(np.arange(0, 105, 20))
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xticklabels(
            [
                model.rstrip("-Instruct-q0f16-MLC").rstrip("-Instruct-q0f32-MLC")
                for model in models
            ],
            rotation=0,
            ha="center",
            fontsize=16,
        )
        if ax_idx == 0:
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(2)
            ]

    # Adjust subplot spacing and title position
    plt.subplots_adjust(
        bottom=0.2, top=0.85, left=0.05, right=0.95, wspace=5, hspace=0.5
    )  # Increased wspace
    fig.legend(
        handles=legend_handles,
        labels=["without structual tag", "with structual tag"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(
        f"{args.summary_root}/accuracy_display.png", dpi=300, bbox_inches="tight"
    )

    # Second figure - Correct schema rate
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))  # Slightly wider figure

    for ax_idx, ax in enumerate(axes.flat):
        dataset = datasets[ax_idx]
        draw_info = {}
        for model in summary:
            draw_info[model] = summary[model][dataset]
        for i, bar in enumerate(bars):
            values = [
                draw_info[model]["correct_schema_rate"][bar] * 100 for model in models
            ]
            ax.bar(x + i * (width + gap), values, width, color=colors[i], label=bar)
        ax.axhline(y=25, color="gray", linestyle="--", linewidth=1)
        ax.axhline(y=50, color="gray", linestyle="--", linewidth=1)
        ax.axhline(y=75, color="gray", linestyle="--", linewidth=1)
        ax.axhline(y=100, color="gray", linestyle="--", linewidth=2)
        ax.set_ylim(0, 105)
        ax.set_xticks(x + width / 2)
        ax.set_title(dataset.replace("_", "-"), fontsize=25, pad=20)  # Added pad
        ax.set_yticks(np.arange(0, 101, 25))
        ax.set_ylabel("Schema Accuracy (%)", fontsize=18)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xticklabels(
            [
                model.rstrip("-Instruct-q0f16-MLC").rstrip("-Instruct-q0f32-MLC")
                for model in models
            ],
            rotation=0,
            ha="center",
            fontsize=16,
        )
        if ax_idx == 0:
            legend_handles = [
                plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(2)
            ]

    plt.subplots_adjust(
        bottom=0.2, top=0.85, left=0.05, right=0.95, wspace=5, hspace=0.3
    )  # Increased wspace
    fig.legend(
        handles=legend_handles,
        labels=["without structual tag", "with structual tag"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.1, 1, 0.9])
    plt.savefig(
        f"{args.summary_root}/correct_schema_rate_display.png",
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Draw")
    parser.add_argument(
        "--summary-root",
        type=str,
        help="The summary root path of the result.",
    )
    args = parser.parse_args()
    with open(f"{args.summary_root}/summary.json", mode="r", encoding="utf-8") as file:
        summary = json.load(file)
    draw(args, summary)
