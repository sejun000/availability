#!/usr/bin/env python3
# opt_config.py  (CSV 다중 입력 지원 버전)
"""
Usage
-----
### 1) 학습 (여러 CSV 합치기)
python opt_config.py train \
    --csv exp1.csv,exp2.csv \
    --features m,k,inter_replicas,intra_replicas,ssd_read_bw,ssd_write_bw \
    --target cost_per_gb \
    --direction minimize \
    --model_out model.pkl

### 2) 탐색
python opt_config.py search \
    --model model.pkl \
    --search_space m:38-43,k:1-6,inter_replicas:0-4,intra_replicas:0-4,ssd_read_bw:5e9-10e9,ssd_write_bw:4e9-8e9 \
    --constraints "m+k==44" \
    --direction minimize \
    --trials 500 --n_jobs 20
"""

import argparse, ast, json, sys
from pathlib import Path

import joblib
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# --------------------------------------------------------------------------- #
# Utils
# --------------------------------------------------------------------------- #
def parse_feature_list(s: str):
    return [f.strip() for f in s.split(",") if f.strip()]


def parse_search_space(s: str):
    ranges = {}
    for token in parse_feature_list(s):
        name, span = token.split(":")
        lo, hi = span.split("-")
        is_int = "." not in lo and "." not in hi
        ranges[name] = (ast.literal_eval(lo), ast.literal_eval(hi), is_int)
    return ranges


def build_optuna_space(trial, ranges):
    params = {}
    for name, (lo, hi, is_int) in ranges.items():
        if is_int:
            params[name] = trial.suggest_int(name, lo, hi)
        else:
            params[name] = trial.suggest_float(name, lo, hi)
    return params


def satisfies_constraints(params: dict, expr: str):
    if not expr:
        return True
    try:
        return bool(eval(expr, {}, params))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def cmd_train(args):
    # 여러 CSV를 쉼표로 받아 모두 concat
    csv_paths = [p.strip() for p in args.csv.split(",") if p.strip()]
    if not csv_paths:
        sys.exit("[train] Error: --csv 인자가 비어 있습니다")

    df_list = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(df_list, ignore_index=True)

    features = parse_feature_list(args.features)
    missing_cols = set(features + [args.target]) - set(df.columns)
    if missing_cols:
        sys.exit(f"[train] Error: CSV에 없는 컬럼: {missing_cols}")

    X, y = df[features], df[args.target]

    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    mae = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error").mean()
    print(f"[train] 5-fold CV MAE = {mae:.4g}")

    model.fit(X, y)
    joblib.dump({"model": model, "features": features}, args.model_out)
    print(f"[train] Saved model → {args.model_out}")


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def cmd_search(args):
    bundle = joblib.load(args.model)
    model, trained_feats = bundle["model"], bundle["features"]

    ranges = parse_search_space(args.search_space)
    unknown = set(ranges) - set(trained_feats)
    if unknown:
        sys.exit(f"[search] Error: 탐색 공간에 학습 안 된 feature 존재: {unknown}")

    direction = "minimize" if args.direction == "minimize" else "maximize"
    study = optuna.create_study(direction=direction)

    def objective(trial):
        params = build_optuna_space(trial, ranges)
        if not satisfies_constraints(params, args.constraints):
            raise optuna.TrialPruned()
        X_new = pd.DataFrame([params])[trained_feats]
        return model.predict(X_new)[0]

    study.optimize(objective, n_trials=args.trials,
                   n_jobs=args.n_jobs, show_progress_bar=True)

    best = study.best_params
    best_val = study.best_value
    print(json.dumps({"best_params": best, "predicted_target": best_val}, indent=2))
    Path("best_params.json").write_text(
        json.dumps({"params": best, "value": best_val}, indent=2))
    print("[search] best_params.json saved")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    p_tr = sub.add_parser("train", help="모델 학습")
    p_tr.add_argument("--csv", required=True,
                      help="쉼표로 구분한 CSV 경로들 (exp1.csv,exp2.csv,...)")
    p_tr.add_argument("--features", required=True,
                      help="학습에 쓸 feature 리스트 (쉼표)")
    p_tr.add_argument("--target", required=True, help="목표 컬럼")
    p_tr.add_argument("--direction", choices=["minimize", "maximize"],
                      default="minimize", help="target 최소/최대화")
    p_tr.add_argument("--model_out", default="model.pkl")

    # search
    p_se = sub.add_parser("search", help="미실험 config 탐색")
    p_se.add_argument("--model", required=True)
    p_se.add_argument("--search_space", required=True,
                      help="f:lo-hi 쉼표 리스트")
    p_se.add_argument("--constraints", default="",
                      help="파라미터 기반 불린식 e.g. 'm+k==44'")
    p_se.add_argument("--direction", choices=["minimize", "maximize"],
                      default="minimize")
    p_se.add_argument("--trials", type=int, default=200)
    p_se.add_argument("--n_jobs", type=int, default=1)

    args = p.parse_args()
    {"train": cmd_train, "search": cmd_search}[args.cmd](args)


if __name__ == "__main__":
    main()
