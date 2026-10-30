"""Retrain RF and matched MLP fusion ablations using one frozen split manifest."""

import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "drafting_code"))
from snooppi_utils import (
    file_sha256,
    load_embedding_cache,
    build_pair_features,
    assert_model_ready,
    require_comparable_preprocessing,
    validate_cache_coverage,
    select_threshold,
    summarize_predictions,
    macro_average_precision,
)
from snooppi_protocol import mapped_overlap_report

ENCODERS = {"prott5": "ProtT5", "esm2": "ESM-2 650M", "esmc": "ESM-C 600M"}
VARIANTS = {
    "full": ("sum", "absdiff", "product"),
    "no_sum": ("absdiff", "product"),
    "no_absdiff": ("sum", "product"),
    "no_product": ("sum", "absdiff"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.splits / "manifest.json").read_text())
    for name, sha in manifest["files"].items():
        if file_sha256(args.splits / name) != sha:
            raise ValueError("Frozen artifact changed: " + name)
    parts = {
        n: pd.read_csv(args.splits / f"{n}.csv")
        for n in ("train", "validation", "test")
    }
    from finish import input_isolation

    input_isolation(parts)
    assert_model_ready(parts)
    require_comparable_preprocessing(parts, list(ENCODERS.values()))
    m = pd.read_csv(args.splits / "mmseqs2_clusters.csv")
    if mapped_overlap_report(parts, m, "cluster_id").overlapping_cluster_id.any():
        raise ValueError("Cluster leakage")
    # Preflight every cache before training any model.
    cache_records = {}
    for k, e in ENCODERS.items():
        c, meta = load_embedding_cache(args.inputs / f"{k}.pt", e)
        validate_cache_coverage(c, parts)
        cache_records[k] = {
            "sha256": file_sha256(args.inputs / f"{k}.pt"),
            "metadata": meta,
        }
        del c
    args.out.mkdir(parents=True, exist_ok=args.resume)
    config = {
        "seed": args.seed,
        "split_manifest_sha256": file_sha256(args.splits / "manifest.json"),
        "caches": cache_records,
        "rf": {"trees": 400, "min_samples_leaf": 2, "max_features": "sqrt"},
        "mlp": {
            "hidden": [512, 512, 256],
            "dropout": 0.1,
            "batch_size": 1024,
            "max_epochs": 50,
            "patience": 10,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "checkpoint": "validation macro AP",
        },
        "variants": VARIANTS,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "python": sys.version,
        "script_sha256": file_sha256(Path(__file__)),
    }
    if (args.out / "manifest.json").exists():
        if json.loads((args.out / "manifest.json").read_text()) != json.loads(
            json.dumps(config)
        ):
            raise ValueError("Resume configuration changed")
    else:
        (args.out / "manifest.json").write_text(json.dumps(config, indent=2))
    y = {n: f.label.to_numpy(dtype=np.float32) for n, f in parts.items()}
    weight = float((y["train"] == 0).sum() / (y["train"] == 1).sum())
    metrics = []
    for k, e in ENCODERS.items():
        cache, _ = load_embedding_cache(args.inputs / f"{k}.pt", e)
        for variant, ops in VARIANTS.items():
            x = {n: build_pair_features(f, cache, ops) for n, f in parts.items()}
            for head in (["rf", "mlp"] if variant == "full" else ["mlp"]):
                start = time.monotonic()
                tag = f"{k}_{head}_{variant}"
                torch.manual_seed(args.seed)
                np.random.seed(args.seed)
                record_path = args.out / f"{tag}_complete.json"
                if args.resume and record_path.exists():
                    record = json.loads(record_path.read_text())
                    for name, sha in record["files"].items():
                        if file_sha256(args.out / name) != sha:
                            raise ValueError("Completed artifact changed")
                    metrics.extend(record["metrics"])
                    continue
                history = []
                best_epoch = None
                if head == "rf":
                    model = RandomForestClassifier(
                        n_estimators=400,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight={0: 1.0, 1: weight},
                        random_state=args.seed,
                        n_jobs=2,
                    )
                    model.fit(x["train"], y["train"])
                    scores = {n: model.predict_proba(v)[:, 1] for n, v in x.items()}
                else:
                    layers = []
                    d = x["train"].shape[1]
                    for h in (512, 512, 256):
                        layers.extend(
                            [
                                torch.nn.Linear(d, h),
                                torch.nn.ReLU(),
                                torch.nn.Dropout(0.1),
                            ]
                        )
                        d = h
                    model = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))
                    optimizer = torch.optim.AdamW(
                        model.parameters(), lr=1e-4, weight_decay=1e-4
                    )
                    loss_fn = torch.nn.BCEWithLogitsLoss(
                        pos_weight=torch.tensor(weight)
                    )
                    xt = {n: torch.from_numpy(v) for n, v in x.items()}
                    yt = torch.from_numpy(y["train"])

                    def predict(n):
                        model.eval()
                        with torch.no_grad():
                            return torch.cat(
                                [
                                    model(z).flatten().sigmoid()
                                    for z in xt[n].split(1024)
                                ]
                            ).numpy()

                    best = -float("inf")
                    stale = 0
                    for epoch in range(1, 51):
                        model.train()
                        order = torch.randperm(len(yt))
                        loss_sum = 0.0
                        for ids in order.split(1024):
                            optimizer.zero_grad()
                            loss = loss_fn(model(xt["train"][ids]).flatten(), yt[ids])
                            loss.backward()
                            optimizer.step()
                            loss_sum += loss.item() * len(ids)
                        ap = macro_average_precision(
                            y["validation"], predict("validation")
                        )
                        history.append(
                            {
                                "epoch": epoch,
                                "train_weighted_loss": loss_sum / len(yt),
                                "validation_macro_ap": ap,
                            }
                        )
                        if ap > best:
                            best = ap
                            state = copy.deepcopy(model.state_dict())
                            best_epoch = epoch
                            stale = 0
                        else:
                            stale += 1
                        if stale >= 10:
                            break
                    model.load_state_dict(state)
                    scores = {n: predict(n) for n in parts}
                    pd.DataFrame(history).to_csv(
                        args.out / f"{tag}_history.csv", index=False
                    )
                threshold = select_threshold(y["validation"], scores["validation"])
                for n in parts:
                    metrics.append(
                        {
                            "encoder": k,
                            "head": head,
                            "variant": variant,
                            "split": n,
                            "seed": args.seed,
                            "best_epoch": best_epoch,
                            "seconds": time.monotonic() - start,
                            **summarize_predictions(y[n], scores[n], threshold),
                        }
                    )
                pd.DataFrame(
                    {
                        "full_pair_id": parts["test"].full_pair_id,
                        "label": y["test"],
                        "probability": scores["test"],
                    }
                ).to_csv(args.out / f"{tag}_test_predictions.csv", index=False)
                files = [args.out / f"{tag}_test_predictions.csv"]
                if history:
                    files.append(args.out / f"{tag}_history.csv")
                record_path.write_text(
                    json.dumps(
                        {
                            "metrics": metrics[-3:],
                            "files": {p.name: file_sha256(p) for p in files},
                        },
                        indent=2,
                    )
                )
                pd.DataFrame(metrics).to_csv(args.out / "metrics.csv", index=False)
                print(
                    tag,
                    "completed",
                    round(time.monotonic() - start, 1),
                    "seconds",
                    flush=True,
                )
                del model
            del x
        del cache
    pd.DataFrame(metrics).to_csv(args.out / "metrics.csv", index=False)
    (args.out / "COMPLETE").write_text(
        "15 fresh classifier fits; one fixed split and matched seed 44.\n"
    )


if __name__ == "__main__":
    main()
