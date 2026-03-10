#!/usr/bin/env python3
import argparse
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "experiments"
LOG_DIR = EXP_DIR / "logs"
LEDGER = EXP_DIR / "ledger.jsonl"
RESULTS_TSV = EXP_DIR / "results.tsv"

VALID_RE = re.compile(r"Valid Loss:\s*([0-9.]+)")
AVG_RE = re.compile(r"On average over 12 horizons, Test MAE:\s*([0-9.]+), Test MAPE:\s*([0-9.]+), Test RMSE:\s*([0-9.]+)")


def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER.touch(exist_ok=True)
    if not RESULTS_TSV.exists():
        RESULTS_TSV.write_text("commit\tvalid_loss\ttest_mae\ttest_mape\ttest_rmse\tduration_sec\tstatus\tdecision\tdescription\n", encoding="utf-8")


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def run_cmd(cmd, cwd=ROOT, check=True, capture=True):
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=capture,
        check=check,
    )


def git_head_short():
    return run_cmd(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()


def git_is_clean():
    out = run_cmd(["git", "status", "--porcelain"]).stdout.strip()
    return out == ""


def git_has_changes(paths=None):
    cmd = ["git", "status", "--porcelain"]
    if paths:
        cmd += ["--", *paths]
    out = run_cmd(cmd).stdout.strip()
    return out != ""


def git_checkout_new_branch(branch: str):
    run_cmd(["git", "checkout", "-b", branch], capture=False)


def git_branch_exists(branch: str):
    p = run_cmd(["git", "branch", "--list", branch])
    return bool(p.stdout.strip())


def git_commit_all(message: str, paths=None):
    add_cmd = ["git", "add"]
    if paths:
        add_cmd += ["--", *paths]
    else:
        add_cmd += ["-A"]
    run_cmd(add_cmd, capture=False)
    run_cmd(["git", "commit", "-m", message], capture=False)


def git_reset_hard(target: str):
    run_cmd(["git", "reset", "--hard", target], capture=False)


def check_git_clean(allow_dirty: bool):
    if not allow_dirty and not git_is_clean():
        raise RuntimeError("git working tree is dirty. commit/stash first or pass --allow-dirty")


def build_cmd(cfg, args):
    cmd = [
        "python", "train.py",
        "--device", args.device,
        "--data", args.data,
        "--adjdata", args.adjdata,
        "--epochs", str(args.epochs),
        "--batch_size", str(cfg["batch_size"]),
        "--nhid", str(cfg["nhid"]),
        "--dropout", str(cfg["dropout"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--weight_decay", str(cfg["weight_decay"]),
        "--adjtype", cfg["adjtype"],
        "--print_every", str(args.print_every),
        "--time_budget_sec", str(args.time_budget_sec),
        "--eval_reserve_sec", str(args.eval_reserve_sec),
        "--min_train_iters", str(args.min_train_iters),
        "--save", str(ROOT / "garage" / f"auto_trial_{cfg['trial_id']}")
    ]
    if cfg.get("gcn_bool", False):
        cmd.append("--gcn_bool")
    if cfg.get("addaptadj", False):
        cmd.append("--addaptadj")
    if cfg.get("randomadj", False):
        cmd.append("--randomadj")
    return cmd


def parse_metrics(text: str):
    valid = VALID_RE.findall(text)
    avg = AVG_RE.findall(text)
    valid_loss = float(valid[-1]) if valid else None
    test_mae = test_mape = test_rmse = None
    if avg:
        test_mae, test_mape, test_rmse = map(float, avg[-1])
    return {
        "valid_loss": valid_loss,
        "test_mae": test_mae,
        "test_mape": test_mape,
        "test_rmse": test_rmse,
    }


def run_trial(cfg, args):
    cmd = build_cmd(cfg, args)
    log_path = LOG_DIR / f"trial_{cfg['trial_id']:03d}.log"
    t0 = time.time()
    status = "ok"
    err_msg = ""

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# cmd\n")
        f.write(" ".join(shlex.quote(x) for x in cmd) + "\n\n")
        f.flush()
        trial_timeout = None
        if args.time_budget_sec > 0:
            trial_timeout = args.time_budget_sec + args.timeout_grace_sec

        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=trial_timeout,
                check=False,
            )
            if proc.returncode != 0:
                status = "error"
                err_msg = f"return_code={proc.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
            err_msg = f"timeout>{trial_timeout}s"

    duration = round(time.time() - t0, 2)
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    metrics = parse_metrics(text)

    if status == "ok" and metrics["valid_loss"] is None:
        status = "error"
        err_msg = "missing valid_loss in log"

    return status, err_msg, duration, metrics, str(log_path)


def make_candidates(n):
    base = {
        "nhid": 32,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 64,
        "adjtype": "doubletransition",
        "gcn_bool": True,
        "addaptadj": True,
        "randomadj": True,
    }
    tweaks = [
        {"nhid": 32, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 1e-4, "batch_size": 64, "gcn_bool": True, "addaptadj": True, "randomadj": True},
        {"nhid": 48, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 1e-4, "batch_size": 64, "gcn_bool": True, "addaptadj": True, "randomadj": True},
        {"nhid": 64, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 1e-4, "batch_size": 32, "gcn_bool": True, "addaptadj": True, "randomadj": True},
        {"nhid": 32, "dropout": 0.2, "learning_rate": 0.001, "weight_decay": 1e-4, "batch_size": 64, "gcn_bool": True, "addaptadj": True, "randomadj": True},
        {"nhid": 32, "dropout": 0.4, "learning_rate": 0.001, "weight_decay": 1e-4, "batch_size": 64, "gcn_bool": True, "addaptadj": True, "randomadj": True},
        {"nhid": 32, "dropout": 0.3, "learning_rate": 0.0005, "weight_decay": 1e-4, "batch_size": 64, "gcn_bool": True, "addaptadj": True, "randomadj": True},
        {"nhid": 32, "dropout": 0.3, "learning_rate": 0.002, "weight_decay": 1e-4, "batch_size": 64, "gcn_bool": True, "addaptadj": True, "randomadj": True},
        {"nhid": 32, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 5e-5, "batch_size": 64, "gcn_bool": True, "addaptadj": True, "randomadj": True},
        {"nhid": 32, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 2e-4, "batch_size": 64, "gcn_bool": True, "addaptadj": True, "randomadj": True},
        {"nhid": 32, "dropout": 0.3, "learning_rate": 0.001, "weight_decay": 1e-4, "batch_size": 64, "gcn_bool": False, "addaptadj": False, "randomadj": False},
    ]

    out = []
    for i in range(n):
        cfg = base.copy()
        cfg.update(tweaks[i % len(tweaks)])
        cfg["adjtype"] = "doubletransition"
        cfg["trial_id"] = i + 1
        out.append(cfg)
    return out


def append_ledger(row):
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_tsv(row):
    line = "\t".join([
        str(row.get("commit", "")),
        str(row.get("valid_loss", "")),
        str(row.get("test_mae", "")),
        str(row.get("test_mape", "")),
        str(row.get("test_rmse", "")),
        str(row.get("duration_sec", "")),
        str(row.get("status", "")),
        str(row.get("decision", "")),
        (row.get("description", "") or "").replace("\t", " "),
    ])
    with open(RESULTS_TSV, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_mutator(mutator_cmd: str):
    proc = subprocess.run(mutator_cmd, cwd=ROOT, shell=True, text=True)
    return proc.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="AutoResearch-GNN orchestrator v0.2")
    parser.add_argument("--mode", choices=["hyper", "code"], default="hyper", help="hyper: parameter-only; code: mutate files + commit + keep/discard rollback")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--time_budget_sec", type=int, default=1200, help="wall-clock budget per trial (training side); 1200 = 20min")
    parser.add_argument("--eval_reserve_sec", type=int, default=180, help="seconds reserved for validation/test before budget cutoff")
    parser.add_argument("--min_train_iters", type=int, default=1, help="minimum train iters before budget stop can trigger")
    parser.add_argument("--timeout_grace_sec", type=int, default=600, help="extra subprocess timeout buffer for eval/test and IO")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--accept_improve", type=float, default=0.005, help="minimum relative improvement for accept")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--setup_branch", type=str, default="", help="create and switch to fresh branch (e.g. autoresearch/mar10)")
    parser.add_argument("--mutator_cmd", type=str, default="", help="shell command to mutate code before each trial (code mode)")
    parser.add_argument("--modifiable_files", type=str, nargs="*", default=["train.py", "model.py"], help="allowed files for code mode commits")
    args = parser.parse_args()

    ensure_dirs()
    check_git_clean(args.allow_dirty)

    if args.setup_branch:
        if git_branch_exists(args.setup_branch):
            raise RuntimeError(f"branch already exists: {args.setup_branch}")
        git_checkout_new_branch(args.setup_branch)

    best_valid = None
    best_trial = None
    candidates = make_candidates(args.trials)

    print(f"[orchestrator] mode={args.mode} trials={args.trials}, epochs={args.epochs}, device={args.device}, time_budget_sec={args.time_budget_sec}")

    for cfg in candidates:
        trial_id = cfg["trial_id"]
        pre_commit = git_head_short()

        if args.mode == "code":
            if not args.mutator_cmd:
                raise RuntimeError("code mode requires --mutator_cmd")
            ok = run_mutator(args.mutator_cmd)
            if not ok:
                raise RuntimeError(f"mutator command failed at trial {trial_id}")
            if not git_has_changes(args.modifiable_files):
                print(f"[trial {trial_id:03d}] mutator made no changes in allowed files, skip")
                continue
            git_commit_all(f"trial {trial_id:03d}: auto mutation", paths=args.modifiable_files)

        run_commit = git_head_short()
        print(f"[trial {trial_id:03d}] commit={run_commit} cfg={cfg}")
        status, err_msg, duration, metrics, log_path = run_trial(cfg, args)

        decision = "reject"
        reason = ""

        if status != "ok":
            reason = err_msg or status
        else:
            val = metrics["valid_loss"]
            if best_valid is None:
                best_valid = val
                best_trial = trial_id
                decision = "accept"
                reason = "first successful baseline"
            else:
                improve = (best_valid - val) / best_valid
                if improve >= args.accept_improve:
                    decision = "accept"
                    reason = f"improved {improve:.4%} >= threshold {args.accept_improve:.2%}"
                    best_valid = val
                    best_trial = trial_id
                else:
                    decision = "reject"
                    reason = f"improved {improve:.4%} < threshold {args.accept_improve:.2%}"

        if args.mode == "code" and decision == "reject":
            git_reset_hard(pre_commit)
            final_commit = git_head_short()
        else:
            final_commit = run_commit

        row = {
            "trial_id": trial_id,
            "timestamp": now_iso(),
            "mode": args.mode,
            "commit_before": pre_commit,
            "commit_run": run_commit,
            "commit_after": final_commit,
            "config": cfg,
            "duration_sec": duration,
            "status": status,
            "metrics": metrics,
            "decision": decision,
            "reason": reason,
            "log_path": log_path,
            "best_valid_so_far": best_valid,
            "best_trial_so_far": best_trial,
        }
        append_ledger(row)
        append_tsv({
            "commit": final_commit,
            "valid_loss": metrics.get("valid_loss"),
            "test_mae": metrics.get("test_mae"),
            "test_mape": metrics.get("test_mape"),
            "test_rmse": metrics.get("test_rmse"),
            "duration_sec": duration,
            "status": status,
            "decision": decision,
            "description": reason,
        })
        print(f"[trial {trial_id:03d}] status={status} decision={decision} valid={metrics.get('valid_loss')} reason={reason} commit_after={final_commit}")

    print("[orchestrator] done")
    if best_valid is not None:
        print(f"[orchestrator] best_valid={best_valid:.4f} at trial={best_trial}")


if __name__ == "__main__":
    main()
