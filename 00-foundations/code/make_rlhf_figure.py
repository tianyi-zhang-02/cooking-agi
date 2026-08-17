"""How many models each preference-optimisation method keeps in memory.

The usual RLHF diagram shows the three stages. It does not show the thing that
decides whether you can run it at all: PPO needs four models resident at once,
and two of them are being trained. That is the number everything since has been
trying to cut.

Run: python make_rlhf_figure.py
"""

import os

from svgkit import svg, text, write

# the note lives in 05-post-training, so the figure belongs beside it
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "05-post-training", "assets")


def rlhf_cost():
    W, H = 730, 372
    CW, BH, GAP = 158, 30, 12
    b = [text(24, 24, "What each method has to keep in memory", "ttl"),
         text(24, 41, "solid = weights being updated · dashed = frozen · "
                      "the trend is downward, and it is not subtle", "sub")]

    METHODS = [
        ("PPO", "InstructGPT, 2022",
         [("Actor", True), ("Critic", True), ("Reward", False), ("Reference", False)],
         "needs rollouts"),
        ("GRPO", "no critic",
         [("Actor", True), ("Reward", False), ("Reference", False)],
         "needs rollouts"),
        ("RLVR", "reward is a program",
         [("Actor", True), ("Reference", False)],
         "needs rollouts"),
        ("DPO", "no RL loop at all",
         [("Policy", True), ("Reference", False)],
         "offline pairs"),
    ]

    for i, (name, sub, models, note) in enumerate(METHODS):
        x = 24 + i * (CW + 14)
        b.append(text(x, 76, name, "ttl"))
        b.append(text(x, 92, sub, "sub"))
        for j, (label, trained) in enumerate(models):
            y = 108 + j * (BH + GAP)
            cls = "box-1" if trained else "box-q"
            b.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{CW}" '
                     f'height="{BH}" rx="6"/>')
            b.append(text(x + CW / 2, y + BH / 2 + 4, label, "lbl", "middle"))
        n_train = sum(t for _, t in models)
        y = 108 + 4 * (BH + GAP) + 6
        b.append(text(x, y, f"{len(models)} models, {n_train} trained", "lbl-s"))
        b.append(text(x, y + 15, note, "lbl-s"))

    b.append(text(24, 330,
                  "PPO's critic is a second full-size network being trained, which is "
                  "why it roughly doubles the trainable", "sub"))
    b.append(text(24, 346,
                  "footprint. GRPO drops it by using the spread within a group of "
                  "samples as the baseline instead.", "sub"))
    b.append(text(24, 362,
                  "RLVR goes further: when the reward is a checkable program, the "
                  "learned reward model disappears too.", "sub"))
    return svg(W, H, "\n".join(b))


if __name__ == "__main__":
    print("drawing the RLHF cost figure...")
    write(OUT, "rlhf-model-count.svg", rlhf_cost())
    print("done.")
