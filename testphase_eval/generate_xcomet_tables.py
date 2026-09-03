"""Build LaTeX tables of average xCOMET scores per team for the Ukrainian and
Sorbian tracks, marking teams that are statistically significantly better
than the TUM_baseline team (paired t-test, p < 0.05, on the paired samples
underlying each results.json file).

Reads: <dir>/<direction>.results.json for each direction below.
Writes: xcomet_ukrainian_table.tex, xcomet_sorbian_table.tex in the same dir.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "xcomet_ukr_results"
BASELINE = "TUM_baseline"
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


def load_direction(direction):
    """Return {team: mean} and {team: significantly_better_than_baseline} for one direction."""
    path = RESULTS_DIR / f"{direction}.results.json"
    data = json.loads(path.read_text())

    means = {}
    better_than_baseline = {}

    for pair in data["results"]:
        x_name, y_name = pair["x_name"], pair["y_name"]
        x_mean = pair["bootstrap_resampling"]["x-mean"]
        y_mean = pair["bootstrap_resampling"]["y-mean"]
        means[x_name] = x_mean
        means[y_name] = y_mean

        if BASELINE not in (x_name, y_name):
            continue
        team = y_name if x_name == BASELINE else x_name
        team_mean = y_mean if x_name == BASELINE else x_mean
        baseline_mean = x_mean if x_name == BASELINE else y_mean
        p_value = pair["paired_t-test"]["p_value"]
        better_than_baseline[team] = p_value < ALPHA and team_mean > baseline_mean

    return means, better_than_baseline


def escape(text):
    return text.replace("_", r"\_")


def format_cell(mean, is_better):
    if mean is None:
        return "--"
    cell = f"{mean * 100:.2f}"
    if is_better:
        cell = r"$^{*}$" + cell
    return cell


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

    col_spec = "l" + "c" * len(directions) + "c"
    header = " & ".join(["Team"] + [escape(d) for d in directions] + ["avg"])

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
            means, better = per_direction[direction]
            mean = means.get(team)
            is_better = better.get(team, False)
            row_cells.append(format_cell(mean, is_better))
            if mean is not None:
                direction_means.append(mean)
        avg = sum(direction_means) / len(direction_means) if direction_means else None
        row_cells.append(format_cell(avg, False))
        lines.append(" & ".join(row_cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{%s $^{*}$~indicates a statistically significant improvement "
        r"over the %s baseline (paired $t$-test, $p < %.2f$).}"
        % (track["caption"], escape(BASELINE), ALPHA)
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
