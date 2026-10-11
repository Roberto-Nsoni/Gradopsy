"""
Write the paper's result tables from the files the training script produced, so no number in the
paper is typed by hand.

    python make_tables.py      # reads ../web/public/recognize/, writes generated/*.tex
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "demos" / "public" / "recognize"
OUT = HERE / "generated"


def pct(v):
    return f"{100 * v:.2f}"


def mnist_table():
    res = json.loads((RESULTS / "results.json").read_text())
    rows = []
    for model, name in (("mlp", "MLP"), ("cnn", "CNN")):
        r = res[model]
        seeds = ", ".join(pct(run["test_accuracy"]) for run in r["runs"])
        minutes = sum(run["seconds"] for run in r["runs"]) / len(r["runs"]) / 60
        rows.append(f"{name} & {r['parameters']:,} & {r['epochs']} & ${pct(r['mean'])} \\pm {pct(r['std'])}$ "
                    f"& {seeds} & ${pct(r['mean_moved'])} \\pm {pct(r['std_moved'])}$ & {minutes:.1f} \\\\")

    return "\n".join([
        "\\begin{table}[H]",
        "\\centering",
        "\\begin{tabular}{@{}lrrrlrr@{}}",
        "\\toprule",
        "model & parameters & epochs & test accuracy (\\%) & per seed & moved test (\\%) & min / run \\\\",
        "\\midrule",
        *rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{MNIST test accuracy, mean $\\pm$ sample standard deviation over three seeds, on the "
        "test set as published and on the same digits moved and resized once with a fixed seed "
        "(moved test). Time is CPU training time in native Python.}",
        "\\label{tab:mnist}",
        "\\end{table}",
    ]) + "\n"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "mnist.tex").write_text(mnist_table())
    print((OUT / "mnist.tex").read_text())
