import argparse
import math
import random

import matplotlib.pyplot as plt


def normalize_token(tok: str) -> str:
    t = tok.strip().upper()
    mapping = {
        "0": "0", "STOP": "0",
        "1": "1", "↑": "1", "FORWARD": "1", "MOVE_FORWARD": "1", "MOVEFORWARD": "1",
        "2": "2", "←": "2", "LEFT": "2", "TURN_LEFT": "2", "TURNLEFT": "2",
        "3": "3", "→": "3", "RIGHT": "3", "TURN_RIGHT": "3", "TURNRIGHT": "3",
    }
    return mapping.get(t, t)


def parse_actions(actions_str: str):
    s = actions_str.strip()
    if "," in s:
        raw = [x for x in s.split(",") if x.strip()]
    elif " " in s:
        raw = [x for x in s.split(" ") if x.strip()]
    else:
        # 允许类似 "112130"
        raw = list(s)

    actions = []
    for r in raw:
        t = normalize_token(r)
        if t in {"0", "1", "2", "3"}:
            actions.append(int(t))
    return actions


def simulate_path(actions, step_len=0.25, turn_deg=15.0):
    # heading=0 表示朝 +Y 方向
    x, y, heading = 0.0, 0.0, 0.0
    points = [(x, y)]
    poses = [(x, y, heading)]

    for a in actions:
        if a == 0:  # STOP
            break
        elif a == 2:  # LEFT
            heading += turn_deg
        elif a == 3:  # RIGHT
            heading -= turn_deg
        elif a == 1:  # FORWARD
            rad = math.radians(heading)
            x += step_len * math.sin(rad)
            y += step_len * math.cos(rad)
            points.append((x, y))
        poses.append((x, y, heading))

    return points, poses


def action_text(actions):
    mp = {0: "STOP", 1: "↑", 2: "←", 3: "→"}
    return " ".join(mp.get(a, str(a)) for a in actions)


def plot_path_on_ax(ax, actions, step_len=0.25, turn_deg=15.0, title_prefix=""):
    points, _ = simulate_path(actions, step_len=step_len, turn_deg=turn_deg)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    if len(points) == 1:
        ax.scatter(xs[0], ys[0], c="tab:blue", s=20)
    else:
        ax.plot(xs, ys, "-o", color="tab:blue", linewidth=2, markersize=3)
        for i in range(len(points) - 1):
            ax.annotate(
                "",
                xy=(xs[i + 1], ys[i + 1]),
                xytext=(xs[i], ys[i]),
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.5),
            )

    ax.scatter(xs[0], ys[0], c="green", s=50, label="START")
    ax.scatter(xs[-1], ys[-1], c="red", s=50, label="END")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{title_prefix}{action_text(actions)}", fontsize=9)


def random_action_seq(min_len=6, max_len=18, stop_prob=0.25):
    n = random.randint(min_len, max_len)
    seq = []
    for _ in range(n):
        if random.random() < stop_prob and len(seq) > 1:
            seq.append(0)
            break
        seq.append(random.choice([1, 2, 3]))
    if seq[-1] != 0:
        seq.append(0)
    return seq


def visualize(actions, out_path="", step_len=0.25, turn_deg=15.0):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    plot_path_on_ax(ax, actions, step_len=step_len, turn_deg=turn_deg)
    fig.suptitle(f"F={step_len}m, Turn={turn_deg}deg", fontsize=11)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=180)
        print(f"saved: {out_path}")
    plt.show()


def visualize_random_subplots(num_seqs=8, out_path="", step_len=0.25, turn_deg=15.0, seed=None):
    if seed is not None:
        random.seed(seed)

    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(num_seqs):
        acts = random_action_seq()
        plot_path_on_ax(
            axes[i],
            acts,
            step_len=step_len,
            turn_deg=turn_deg,
            title_prefix=f"Seq {i + 1}: ",
        )

    for j in range(num_seqs, rows * cols):
        axes[j].axis("off")

    fig.suptitle("Random Action Sequences (8 subplots)", fontsize=13)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=180)
        print(f"saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 4-action sequence as rough path.")
    parser.add_argument("--actions", type=str, default="",
                        help='e.g. "1,1,2,1,0" or "↑ ↑ ← ↑ STOP" or "11210"')
    parser.add_argument("--out", type=str, default="",
                        help="optional output image path")
    parser.add_argument("--step_len", type=float, default=0.25)
    parser.add_argument("--turn_deg", type=float, default=15.0)
    parser.add_argument("--random_test", action="store_true",
                        help="show 8 random action sequences in one figure")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.random_test:
        visualize_random_subplots(
            num_seqs=8,
            out_path=args.out,
            step_len=args.step_len,
            turn_deg=args.turn_deg,
            seed=args.seed,
        )
    else:
        acts = parse_actions(args.actions)
        if not acts:
            raise ValueError("No valid actions parsed. Use --actions with 0/1/2/3 or STOP/↑/←/→, or use --random_test.")
        visualize(acts, args.out, step_len=args.step_len, turn_deg=args.turn_deg)