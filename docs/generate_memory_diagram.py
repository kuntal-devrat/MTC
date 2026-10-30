#!/usr/bin/env python3
"""
Generates high-resolution diagram: Ping-Pong Delta Buffering & Fused In-Place Parameter Updates
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Background styling
ax.add_patch(patches.Rectangle((0, 0), 100, 100, facecolor="#ffffff", edgecolor="none"))

# Title
ax.text(50, 95, "MTC Reverse-Mode Memory Schedule: Ping-Pong Buffering & In-Place SGD",
        ha="center", va="center", fontsize=12.5, fontweight="bold", color="#0f172a")

# Naive Autograd Graph Box (Left)
ax.add_patch(patches.FancyBboxPatch((3, 10), 44, 78, boxstyle="round,pad=1.5",
                                    facecolor="#f8fafc", edgecolor="#94a3b8", linewidth=1.2))
ax.text(25, 83, "Traditional Autograd Engine (e.g., PyTorch / AIfES)",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#dc2626")

naive_boxes = [
    ("Dynamic Heap Graph (Tape)", "#fee2e2", "#ef4444", 72, "Requires runtime malloc() / free()"),
    ("Dedicated Parameter Gradients (∇W, ∇b)", "#fee2e2", "#ef4444", 57, "Memory duplicated: sizeof(∇W) = sizeof(W)"),
    ("Retained Activation Errors (δ_1 ... δ_L)", "#fee2e2", "#ef4444", 42, "One dedicated array per layer"),
    ("Post-Graph Optimizer Step", "#fee2e2", "#ef4444", 27, "Weights updated only after entire backward pass"),
]

for title, bg, border, y, desc in naive_boxes:
    ax.add_patch(patches.FancyBboxPatch((6, y-4), 38, 9, boxstyle="round,pad=0.8",
                                        facecolor=bg, edgecolor=border, linewidth=1))
    ax.text(25, y+1.5, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#7f1d1d")
    ax.text(25, y-2, desc, ha="center", va="center", fontsize=7.2, color="#450a0a")

ax.text(25, 14, "Dynamic heap causes fragmentation; 100% parameter memory bloat",
        ha="center", va="center", fontsize=7.8, fontweight="bold", color="#b91c1c")

# MTC Static Planner Box (Right)
ax.add_patch(patches.FancyBboxPatch((53, 10), 44, 78, boxstyle="round,pad=1.5",
                                    facecolor="#f0fdf4", edgecolor="#16a34a", linewidth=1.5))
ax.text(75, 83, "MTC Ahead-of-Time Static Architecture",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#15803d")

mtc_boxes = [
    ("Static Memory Pool (malloc = 0)", "#dcfce7", "#22c55e", 72, "Single fixed array in .bss at compile time"),
    ("Fused In-Place SGD (No ∇W, No ∇b)", "#dcfce7", "#22c55e", 57, "W ← W - η(δ · A^T) computed in-place immediately"),
    ("Alternating Ping-Pong Buffers", "#dcfce7", "#22c55e", 42, "Buffer A ⇄ Buffer B: only 2 scratchpads needed"),
    ("Single-Pass Reverse-Mode Execution", "#dcfce7", "#22c55e", 27, "Blame propagated back before weights are touched"),
]

for title, bg, border, y, desc in mtc_boxes:
    ax.add_patch(patches.FancyBboxPatch((56, y-4), 38, 9, boxstyle="round,pad=0.8",
                                        facecolor=bg, edgecolor=border, linewidth=1.2))
    ax.text(75, y+1.5, title, ha="center", va="center", fontsize=8.5, fontweight="bold", color="#14532d")
    ax.text(75, y-2, desc, ha="center", va="center", fontsize=7.2, color="#14532d")

ax.text(75, 14, "Deterministic static layout; 40.1% peak SRAM reduction; zero malloc",
        ha="center", va="center", fontsize=7.8, fontweight="bold", color="#15803d")

# Central Arrow
ax.annotate("", xy=(52, 50), xytext=(48, 50),
            arrowprops=dict(arrowstyle="->", lw=2, color="#0b69a3"))

plt.tight_layout()
plt.savefig("docs/images/ping_pong_fused_liveness.png", dpi=300, bbox_inches="tight")
print("Saved docs/images/ping_pong_fused_liveness.png")
