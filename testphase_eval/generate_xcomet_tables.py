"""Build LaTeX tables of average xCOMET scores per team for the Ukrainian and
Sorbian tracks, with a WMT-style significance-cluster rank per direction:
rank(team) = 1 + number of other teams whose mean is significantly higher
(paired t-test, p < 0.05, Holm-Bonferroni corrected within each direction's
family of pairwise tests).

Reads: <dir>/<direction>.results.json for each direction below.
Writes: xcomet_ukrainian_table.tex, xcomet_sorbian_table.tex in the same dir.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "xcomet_ukr_results"
ALPHA = 0.05

TRACKS = {
    "ukrainian": {
        "directions": ["cs-uk", "en-uk"],
        "caption": "Average xCOMET scores for the Ukrainian track.",
        "label": "tab:xcomet-ukrainian",
    },
    "sorbian": {
        "directions": ["hsb-de", "dsb-de"],
        "caption": "Average xCOMET scores for the Sorbian track.",
        "label": "tab:xcomet-sorbian",
    },
}


def holm_bonferroni(p_values, alpha=ALPHA):
    """Step-down Holm-Bonferroni correction. Returns a bool array of rejections."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    reject = [False] * m
    for rank, i in enumerate(order):
        if p_values[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break  # step-down: stop at first non-rejection
    return reject


def load_direction(direction):
    """Return {team: mean} and {team: rank} for one direction.

    rank(team) = 1 + number of other teams that are significantly better
    (paired t-test, Holm-Bonferroni corrected p < ALPHA, and higher mean)
    than team, correcting across all pairwise tests run for this direction.
    """
    path = RESULTS_DIR / f"{direction}.results.json"
    data = json.loads(path.read_text())

    means = {}
    beaten_by = {}  # team -> set of teams significantly better than it

    pairs = data["results"]
    p_values = [pair["paired_t-test"]["p_value"] for pair in pairs]
    significant = holm_bonferroni(p_values)

    for pair, is_significant in zip(pairs, significant):
        x_name, y_name = pair["x_name"], pair["y_name"]
        x_mean = pair["bootstrap_resampling"]["x-mean"]
        y_mean = pair["bootstrap_resampling"]["y-mean"]
        means[x_name] = x_mean
        means[y_name] = y_mean
        beaten_by.setdefault(x_name, set())
        beaten_by.setdefault(y_name, set())

        if not is_significant:
            continue
        if x_mean > y_mean:
            beaten_by[y_name].add(x_name)
        elif y_mean > x_mean:
            beaten_by[x_name].add(y_name)

    ranks = {team: 1 + len(better) for team, better in beaten_by.items()}
    return means, ranks


def escape(text):
    return text.replace("_", r"\_")


def format_score(mean):
    if mean is None:
        return "--"
    return f"{mean * 100:.2f}"


def format_rank(rank):
    return "--" if rank is None else str(rank)


def build_table(track_name, track):
    directions = track["directions"]
    per_direction = {d: load_direction(d) for d in directions}

    teams = set()
    for means, _ in per_direction.values():
        teams.update(means)

    def avg_for(team):
        vals = [means[team] for means, _ in per_direction.values() if team in means]
        return sum(vals) / len(vals) if vals else float("-inf")

    ordered_teams = sorted(teams, key=avg_for, reverse=True)

    col_spec = "l" + "cc" * len(directions) + "c"
    header_cols = ["Team"]
    for d in directions:
        header_cols.append(escape(d))
        header_cols.append("rank")
    header_cols.append("avg")
    header = " & ".join(header_cols)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{%s}" % col_spec)
    lines.append(r"\toprule")
    lines.append(header + r" \\")
    lines.append(r"\midrule")

    for team in ordered_teams:
        row_cells = [escape(team)]
        direction_means = []
        for direction in directions:
            means, ranks = per_direction[direction]
            mean = means.get(team)
            rank = ranks.get(team)
            row_cells.append(format_score(mean))
            row_cells.append(format_rank(rank))
            if mean is not None:
                direction_means.append(mean)
        avg = sum(direction_means) / len(direction_means) if direction_means else None
        row_cells.append(format_score(avg))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{%s The rank columns give the significance-cluster rank "
        r"per direction: rank $k$ means $k-1$ teams are significantly "
        r"better (paired $t$-test, $p < %.2f$, Holm-Bonferroni corrected "
        r"within each direction); tied ranks are not significantly "
        r"different from each other.}"
        % (track["caption"], ALPHA)
    )
    lines.append(r"\label{%s}" % track["label"])
    lines.append(r"\end{table}")
    lines.append("")

    return "\n".join(lines)


def main():
    for track_name, track in TRACKS.items():
        table = build_table(track_name, track)
        out_path = RESULTS_DIR / f"xcomet_{track_name}_table.tex"
        out_path.write_text(table)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
