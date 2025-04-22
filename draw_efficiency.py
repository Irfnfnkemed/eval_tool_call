import argparse
import json
import pandas as pd
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

models = [
    [
        "Llama-3.2-1B-Instruct-q0f16-MLC",
        "Llama-3.2-3B-Instruct-q0f16-MLC",
        "Llama-3.1-8B-Instruct-q0f16-MLC",
    ],
    ["Llama-3.2-1B-Instruct", "Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct"],
]
dataset = "BFCL_v3_multiple"
query_to_title = {
    "end_to_end_latency_s.mean": "Average end-to-end latency (s)",
    "time_per_output_token_s.mean": "Average time per output token (s)",
    "time_to_first_token_s.mean": "Average time to first token (s)",
    "output_tokens.mean": "Average output tokens",
}


def draw(args, mlc: Dict, sglang: Dict, query: str):
    colors = ["#0056b3", "#FF8C00"]
    summary = [mlc, sglang]

    # First figure - MLCLLM
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Slightly wider figure
    bars = ["no_stag", "use_stag_jf"]
    width = 0.35
    gap = 0.03
    x = np.arange(len(models[0]))

    for ax_idx, ax in enumerate(axes.flat):
        draw_info = [{}, {}]
        for model in summary[ax_idx]:
            draw_info[ax_idx][model] = summary[ax_idx][model][dataset]
        for i, bar in enumerate(bars):
            values = [draw_info[ax_idx][model][bar][query] for model in models[ax_idx]]
            ax.bar(x + i * (width + gap), values, width, color=colors[i], label=bar)
        ax.set_title(["MLC-LLM", "SGLang"][ax_idx], fontsize=25, pad=20)  # Added pad
        ax.set_xticks(x + width / 2)
        ax.set_ylabel(query_to_title[query], fontsize=18)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xticklabels(
            [
                model.rstrip("-Instruct-q0f16-MLC").rstrip("-Instruct-q0f32-MLC")
                for model in models[ax_idx]
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
    plt.savefig(f"{args.bench_root}/{query}.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Draw")
    parser.add_argument(
        "--bench-root",
        type=str,
        help="The bench root path of the result.",
    )
    args = parser.parse_args()
    with open(f"{args.bench_root}/mlc/bench.json", mode="r", encoding="utf-8") as file:
        mlc = json.load(file)
    with open(
        f"{args.bench_root}/sglang/bench.json", mode="r", encoding="utf-8"
    ) as file:
        sglang = json.load(file)
    draw(args, mlc, sglang, "output_tokens.mean")
    # draw(args, mlc, sglang, "time_per_output_token_s.mean")
    # draw(args, mlc, sglang, "time_to_first_token_s.mean")
