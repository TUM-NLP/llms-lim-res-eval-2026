"""Score Ukrainian MT submissions (cs-uk, en-uk) with XCOMET and test pairwise significance.

Reuses `wmt_st_submission_extraction.extract_file`/`extract_subtask` to read
submission files in the shared-task layout instead of expecting flat
--pred/--pred2/--ref jsonl paths. For every MT subtask, every team's
prediction file is scored once and all pairwise comparisons are derived from
that single scored array (bootstrap_resampling/pairwise_bootstrap already
handle N systems), rather than rescoring per pair.
"""
import argparse
import itertools
import json
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pandas as pd
from comet import download_model, load_from_checkpoint
from comet.models import CometModel
from comet.cli.compare import (
    bootstrap_resampling,
    display_statistical_results,
    pairwise_bootstrap,
    t_tests_summary,
)
from comet.models.utils import Prediction
from scipy import stats

from wmt_st_submission_extraction import TASK_GOLD_DICT, extract_file, extract_subtask

# Only the Ukrainian MT subtasks are in scope for this script.
UKR_MT_TASK = "ukr-mt"
UKR_MT_SUBTASKS = ("cs-uk", "en-uk")


def parse_args() -> argparse.Namespace:
    """Parse and return the CLI arguments for this script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teams-json", type=str, required=True,
        help="Path to wmt2026_teams.json listing each team's primary submissions.",
    )
    parser.add_argument(
        "--submissions-dir", type=str, required=True,
        help="Parent folder of the Ukrainian submission files.",
    )
    parser.add_argument(
        "--gold-dir", type=str, required=True,
        help="Parent folder of the Ukrainian MT gold test files.",
    )
    parser.add_argument(
        "--comet-model", type=str, default="Unbabel/XCOMET-XXL",
        help="COMET model name/checkpoint id to download via comet.download_model.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="xcomet_ukr_results",
        help="Directory to write per-subtask error-span logs and score summaries to.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Skip downloading/running the COMET model (no GPU needed). Still reads "
            "all submission/gold files and exercises the full pipeline, but scores "
            "are random placeholders, not real COMET output."
        ),
    )
    return parser.parse_args()


def build_submission_dict(teams_json_path: str) -> dict[str, dict[str, str]]:
    """Map each team name to its {task: file_name} dict, via extract_file."""
    wmt_teams_df = pd.read_json(path_or_buf=teams_json_path)
    return {
        wmt_teams_df.name[i]: extract_file(wmt_teams_df.primary_submissions[i])
        for i in range(len(wmt_teams_df))
    }


def load_team_ukr_mt_predictions(
    team_name: str,
    submission_dict: dict[str, dict[str, str]],
    submissions_dir: str,
) -> Optional[dict[str, pd.DataFrame]]:
    """Read one team's Ukrainian MT file and split it into {subtask: df} via extract_subtask.

    Returns None if the team did not submit the Ukrainian MT task.
    """
    file_name = submission_dict[team_name].get(UKR_MT_TASK)
    if file_name is None:
        return None
    file_path = os.path.join(submissions_dir, file_name)
    aggregated_df = pd.read_json(path_or_buf=file_path, lines=True)
    return extract_subtask(aggregated_df)


def load_gold_ukr_mt(gold_dir: str) -> dict[str, pd.DataFrame]:
    """Read the cs-uk and en-uk gold files, keyed by subtask name.

    Gold file names are taken from TASK_GOLD_DICT (basename only) so the
    expected naming convention lives in one place, joined with the
    caller-supplied gold_dir instead of the hard-coded folder in that dict.
    """
    gold = {}
    for subtask in UKR_MT_SUBTASKS:
        file_name = os.path.basename(TASK_GOLD_DICT[subtask])
        gold[subtask] = pd.read_json(os.path.join(gold_dir, file_name), lines=True)
    return gold


def build_comet_input(
    pred_df: pd.DataFrame, gold_df: pd.DataFrame, src_lang: str
) -> list[dict[str, str]]:
    """Merge a team's predictions with gold on sent_id and format for COMET.

    src_lang is the gold column holding the source-language text ('cs' or
    'en'); the reference is always the 'ukr' column.
    """
    merged = pred_df.merge(gold_df, on="sent_id", how="inner", validate="one_to_one")
    assert len(merged) == len(gold_df), (
        f"sent_id mismatch between predictions and gold: {len(merged)} vs {len(gold_df)}"
    )
    return [
        {"src": row[src_lang], "mt": row["pred"], "ref": row["ukr"]}
        for _, row in merged.iterrows()
    ]


def make_dummy_prediction(num_segments: int) -> SimpleNamespace:
    """Build a Prediction-shaped stand-in with random scores, for --dry-run.

    Not an actual comet.models.utils.Prediction (its constructor isn't
    something to guess at); a SimpleNamespace exposing the same subset of
    attributes this script reads (.scores, .system_score,
    .metadata.error_spans) so log_error_spans/print_score_summary/
    run_pairwise_significance don't need a dry-run branch of their own.
    """
    scores = list(np.random.uniform(0.0, 1.0, size=num_segments))
    return SimpleNamespace(
        scores=scores,
        system_score=float(np.mean(scores)),
        metadata=SimpleNamespace(error_spans=[[] for _ in range(num_segments)]),
    )


def score_team_submissions(
    model: Optional[CometModel],
    team_inputs: dict[str, list[dict[str, str]]],
    batch_size: int,
    gpus: int,
    dry_run: bool = False,
) -> dict[str, Prediction | SimpleNamespace]:
    """Run the COMET model once per team and return {team_name: model_output}.

    In --dry-run mode, no model is required; random placeholder predictions
    are returned instead so the rest of the pipeline can be exercised.
    """
    if dry_run:
        return {team_name: make_dummy_prediction(len(comet_input)) for team_name, comet_input in team_inputs.items()}
    return {
        team_name: model.predict(comet_input, batch_size=batch_size, gpus=gpus)
        for team_name, comet_input in team_inputs.items()
    }


def log_error_spans(
    model_outputs: dict[str, Prediction | SimpleNamespace], subtask: str, output_dir: str
) -> None:
    """Write one error-span log file per team for a given subtask."""
    for team_name, model_output in model_outputs.items():
        log_path = os.path.join(output_dir, f"{subtask}.{team_name}.errorspans.log")
        with open(log_path, "w") as f:
            for error_spans in model_output.metadata.error_spans:
                f.write(f"{error_spans}\n")


def print_score_summary(
    model_outputs: dict[str, Prediction | SimpleNamespace], subtask: str, dry_run: bool = False
) -> None:
    """Print each team's system-level COMET score for one subtask, best first."""
    ranked = sorted(model_outputs.items(), key=lambda kv: kv[1].system_score, reverse=True)
    header = f"[{subtask}] system-level XCOMET scores"
    if dry_run:
        header = f"*** DRY RUN *** {header} (random placeholders, not real COMET output)"
    print(f"{header}:")
    for team_name, model_output in ranked:
        print(f"  {team_name}: {model_output.system_score:.4f}")


@dataclass(frozen=True)
class SystemName:
    """Duck-typed Path_fr stand-in: comet.cli.compare's helpers read `.rel_path`
    off each system identifier (built for comparing files); we compare teams."""

    rel_path: str

    def __str__(self) -> str:
        return self.rel_path


def run_pairwise_significance(
    model_outputs: dict[str, Prediction | SimpleNamespace], subtask: str
) -> Optional[list[dict]]:
    """Bootstrap-resample and paired-t-test every pair of teams for one subtask.

    Requires at least two teams; returns None otherwise. Segment scores for
    all teams are scored once upstream (score_team_submissions), so this
    only computes the pairwise statistics, not fresh COMET predictions. Works
    the same on --dry-run's random placeholder scores.
    """
    team_names = list(model_outputs.keys())
    if len(team_names) < 2:
        print(f"[{subtask}] fewer than 2 teams submitted, skipping significance test")
        return None

    system_names = [SystemName(name) for name in team_names]
    seg_scores = np.array([model_outputs[name].scores for name in team_names], dtype="float32")
    population_size = seg_scores.shape[1]
    sys_scores = bootstrap_resampling(
        seg_scores,
        sample_size=max(int(population_size * 0.4), 1),
        num_splits=300,
    )
    results = list(pairwise_bootstrap(sys_scores, system_names))

    pairs = itertools.combinations(zip(system_names, seg_scores), 2)
    for (x_name, x_seg_scores), (y_name, y_seg_scores) in pairs:
        ttest_result = stats.ttest_rel(x_seg_scores, y_seg_scores, alternative="two-sided")
        for res in results:
            if res["x_name"] == x_name and res["y_name"] == y_name:
                res["paired_t-test"] = {
                    "statistic": ttest_result.statistic,
                    "p_value": ttest_result.pvalue,
                }

    for res in results:
        display_statistical_results(res)
    print()
    t_tests_summary(results, tuple(system_names))
    return results


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dry_run:
        print("*** DRY RUN *** no COMET model will be downloaded or run; scores are random placeholders.")
        model = None
    else:
        checkpoint_path = download_model(args.comet_model)
        model = load_from_checkpoint(checkpoint_path)

    submission_dict = build_submission_dict(args.teams_json)
    gold = load_gold_ukr_mt(args.gold_dir)

    for subtask in UKR_MT_SUBTASKS:
        src_lang = subtask.split("-")[0]
        gold_df = gold[subtask]

        team_inputs = {}
        for team_name in submission_dict:
            subtask_dfs = load_team_ukr_mt_predictions(team_name, submission_dict, args.submissions_dir)
            if subtask_dfs is None or subtask not in subtask_dfs:
                continue
            team_inputs[team_name] = build_comet_input(subtask_dfs[subtask], gold_df, src_lang)

        print(f"[{subtask}] scoring {len(team_inputs)} teams: {list(team_inputs)}")
        model_outputs = score_team_submissions(
            model, team_inputs, args.batch_size, args.gpus, dry_run=args.dry_run
        )
        log_error_spans(model_outputs, subtask, args.output_dir)
        print_score_summary(model_outputs, subtask, dry_run=args.dry_run)

        results = run_pairwise_significance(model_outputs, subtask)
        if results is not None:
            summary_path = os.path.join(args.output_dir, f"{subtask}.results.json")
            with open(summary_path, "w") as f:
                json.dump({"dry_run": args.dry_run, "results": results}, f, default=str, indent=2)


if __name__ == "__main__":
    main()
