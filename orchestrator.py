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

VALID_RE = re.compile(r"Valid Loss:\s*([0-9.]+)")
AVG_RE = re.compile(r"On average over 12 horizons, Test MAE:\s*([0-9.]+), Test MAPE:\s*([0-9.]+), Test RMSE:\s*([0-9.]+)")


def ensure_dirs():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER.touch(exist_ok=True)


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def check_git_clean(allow_dirty: bool):
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    dirty = bool(proc.stdout.strip())
    if dirty and not allow_dirty:
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
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
                check=False,
            )
            if proc.returncode != 0:
                status = "error"
                err_msg = f"return_code={proc.returncode}"
        except subprocess.TimeoutExpired:
            status = "timeout"
            err_msg = f"timeout>{args.timeout}s"

    duration = round(time.time() - t0, 2)
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    metrics = parse_metrics(text)

    if status == "ok" and metrics["valid_loss"] is None:
        status = "error"
        err_msg = "missing valid_loss in log"

    return status, err_msg, duration, metrics, str(log_path)


def make_candidates(n):
    # 手工低风险搜索空间（可复现）
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


def main():
    parser = argparse.ArgumentParser(description="AutoResearch-GNN orchestrator v0.1")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--accept_improve", type=float, default=0.005, help="minimum relative improvement for accept")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    check_git_clean(args.allow_dirty)

    best_valid = None
    best_trial = None
    candidates = make_candidates(args.trials)

    print(f"[orchestrator] start trials={args.trials}, epochs={args.epochs}, device={args.device}")

    for cfg in candidates:
        print(f"[trial {cfg['trial_id']:03d}] cfg={cfg}")
        status, err_msg, duration, metrics, log_path = run_trial(cfg, args)

        decision = "reject"
        reason = ""

        if status != "ok":
            reason = err_msg or status
        else:
            val = metrics["valid_loss"]
            if best_valid is None:
                best_valid = val
                best_trial = cfg["trial_id"]
                decision = "accept"
                reason = "first successful baseline"
            else:
                improve = (best_valid - val) / best_valid
                if improve >= args.accept_improve:
                    decision = "accept"
                    reason = f"improved {improve:.4%} >= threshold {args.accept_improve:.2%}"
                    best_valid = val
                    best_trial = cfg["trial_id"]
                else:
                    decision = "reject"
                    reason = f"improved {improve:.4%} < threshold {args.accept_improve:.2%}"

        row = {
            "trial_id": cfg["trial_id"],
            "timestamp": now_iso(),
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
        print(f"[trial {cfg['trial_id']:03d}] status={status} decision={decision} valid={metrics.get('valid_loss')} reason={reason}")

    print("[orchestrator] done")
    if best_valid is not None:
        print(f"[orchestrator] best_valid={best_valid:.4f} at trial={best_trial}")


if __name__ == "__main__":
    main()
