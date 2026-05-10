#!/usr/bin/env python3
"""Simple plotting helper for eval JSONs created by the evaluation scripts.
Usage: python scripts/plot_results.py results/imagenet_mini_knn_results.json results/imagenet_mini_plot.png
"""
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input_json')
    p.add_argument('out_png')
    args = p.parse_args()

    with open(args.input_json, 'r') as f:
        data = json.load(f)

    backends = list(data.keys())
    top1 = [data[b]['top1_acc'] * 100.0 for b in backends]
    top5 = [data[b]['top5_acc'] * 100.0 for b in backends]
    latency = [data[b]['mean_latency_ms_per_image'] for b in backends]

    x = np.arange(len(backends))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10,4))

    ax = axes[0]
    ax.bar(x - width/2, top1, width, label='Top-1')
    ax.bar(x + width/2, top5, width, label='Top-5')
    ax.set_xticks(x)
    ax.set_xticklabels(backends)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('k-NN Accuracy (leave-one-out)')
    ax.legend()

    ax2 = axes[1]
    ax2.bar(x, latency, color='C2')
    ax2.set_xticks(x)
    ax2.set_xticklabels(backends)
    ax2.set_ylabel('Mean latency (ms/image)')
    ax2.set_title('Mean latency per image')

    plt.tight_layout()
    plt.savefig(args.out_png)
    print('Wrote', args.out_png)

if __name__ == '__main__':
    main()
