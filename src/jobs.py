# -*- coding: utf-8 -*-
"""
src/jobs.py — Background execution for the long "full comparison" benchmark.

Why a background thread:
  * the UI stays responsive — you can browse other pages while it runs (the
    thread is not tied to Streamlit's per-script execution, so navigating away
    does not kill it);
  * the classic models (CPU: sklearn / XGBoost) and the CNN (GPU: PyTorch) run
    in PARALLEL, so wall-clock time ≈ max(classic, CNN) instead of the sum.

The compute functions never call ``st`` and receive already-fetched data, so
they are safe to run off the Streamlit ScriptRunContext.
"""

import concurrent.futures as _cf
import contextlib
import io
import json
import math
import os
import threading
from statistics import mean as _mean, stdev as _stdev
from typing import Callable, Dict, List, Optional, Tuple

import joblib

from src.data_loader import (
    read_attack_ids, split_unseen_attacks, stratified_subsample,
)
from src.pipeline import (
    MODEL_OPTIONS, extract_feature_matrix, run_classic_models,
    score_fitted_classic, evaluate_cnn_on_set, evaluate_raw_on_set,
    train_and_evaluate_cnn,
)
from src.reporting import (
    COL_ACCURACY, COL_EER, COL_FEATURES, COL_FEAT_TIME, COL_INFER_TIME,
    COL_MIN_DCF, COL_MODEL, COL_THRESHOLD, COL_TRAIN_TIME,
)

FEATURE_ORDER = ["1", "2", "3", "4", "6"]

# One job at a time; persists across Streamlit reruns (module-level).
_pool = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="bench")

# Live progress shared with the UI (the worker can't call st). Read via progress().
_lock  = threading.Lock()
_state = {"total": 0, "done": 0, "label": "Starting…"}

# Per-stream progress so the leaderboard can show classic (CPU) and CNN (GPU)
# side by side while they run in parallel.
def _blank_stream():
    return {"done": 0, "total": 0, "label": "Waiting…", "items": []}


_streams = {"classic": _blank_stream(), "cnn": _blank_stream()}

# Cooperative cancellation: the UI sets this; the workers check it at safe
# checkpoints (between extractors / before each CNN epoch) and stop early.
_cancel = threading.Event()


def request_cancel() -> None:
    _cancel.set()


def cancel_requested() -> bool:
    return _cancel.is_set()


def progress() -> Dict:
    """Snapshot of the running job's progress (overall + per-stream)."""
    with _lock:
        total = max(_state["total"], 1)
        streams = {k: {**v, "items": list(v["items"])} for k, v in _streams.items()}
        return {"frac": _state["done"] / total, "done": _state["done"],
                "total": _state["total"], "label": _state["label"],
                "streams": streams}


def _reset_streams(classic_total: int, cnn_total: int) -> None:
    with _lock:
        _streams["classic"] = _blank_stream(); _streams["classic"]["total"] = classic_total
        _streams["cnn"]     = _blank_stream(); _streams["cnn"]["total"]     = cnn_total


def _stream(name: str, label=None, inc: int = 0, item: str = None) -> None:
    with _lock:
        s = _streams[name]
        if label is not None:
            s["label"] = label
        s["done"] += inc
        if item:
            s["items"].append(item)


def _step(label: str, inc: int = 0) -> None:
    with _lock:
        _state["done"] += inc
        _state["label"] = label


# Live per-epoch records for a background CNN training (worker appends, UI polls).
_cnn_epochs: List[Dict] = []


def cnn_epochs() -> List[Dict]:
    """Snapshot of the epoch records produced so far by a background training."""
    with _lock:
        return [dict(r) for r in _cnn_epochs]


def _run_cnn(train_samples, dev_samples, extractor, params,
             eval_samples=None, eval_sets=None):
    """Background CNN training. Mirrors each epoch into the shared progress so the
    UI can draw a live loss curve and a progress bar without touching the worker."""
    _cancel.clear()
    max_ep = int(params.get("epochs", 1))
    with _lock:
        _cnn_epochs.clear()
        _state.update({"total": max_ep, "done": 0, "label": "Training CNN…"})

    def _cb(rec):
        with _lock:
            _cnn_epochs.append(dict(rec))
            _state["done"]  = int(rec.get("epoch", _state["done"]))
            _state["label"] = (f"Epoch {rec.get('epoch')}/{max_ep} · "
                               f"val={rec.get('val_loss', 0):.4f}")

    def _cb_batch(epoch, n_batch, loss):
        with _lock:
            _state["label"] = (f"Epoch {epoch}/{max_ep} · "
                               f"batch {n_batch} · loss={loss:.4f}")

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        model, history, results = train_and_evaluate_cnn(
            train_samples, dev_samples, extractor, params,
            eval_samples=eval_samples, eval_sets=eval_sets, epoch_callback=_cb,
            batch_callback=_cb_batch, should_stop=_cancel.is_set,
            checkpoint_path="asvspoof_model_checkpoint.pth",
        )
    return model, history, results


def submit_cnn_training(**kwargs) -> "_cf.Future":
    """Launch a single CNN training in the background; returns a Future of
    (model, history, results). Reuses the one-job pool so it never collides with
    a full comparison (the UI already forbids starting two jobs at once)."""
    return _pool.submit(_run_cnn, **kwargs)


def _tag_split(results: List[Dict], base_split: str) -> List[Dict]:
    out = []
    for r in results:
        r = dict(r)
        if "[EVAL]" in str(r.get(COL_MODEL, "")):
            r[COL_MODEL]    = r[COL_MODEL].replace("[EVAL]", "").strip()
            r[COL_FEATURES] = str(r.get(COL_FEATURES, "")).replace("[EVAL]", "").strip()
            _c = str(r.get("Corpus", "")).strip()
            r["Split"]      = f"eval · {_c}" if _c else "eval"
        else:
            r["Split"] = base_split
        out.append(r)
    return out


# ===========================================================================
# Demo-model export (Full comparison, LOCAL)
# ===========================================================================
# As the full sweep runs locally it also PERSISTS the registry models (the two
# CNNs as .pth, the XGBoost × DSP detectors as .joblib) and records their
# dev/eval EER & minDCF, so the leaderboard the cloud demo serves is produced
# straight from the UI — no external console script needed.
_leaderboard: Dict[str, Dict] = {}

# Distinctive token in each classifier's COL_MODEL display name, for picking its
# rows out of run_classic_models' results (which trains all three per feature).
_CLF_TOKEN = {"logistic_regression": "Logistic", "svm_lineal": "SVM",
              "xgboost": "XGBoost"}


def _registry():
    """Lazy import of the model registry (defers the Streamlit-heavy ui_helpers
    import out of module load and worker threads)."""
    from src.ui_helpers import (
        LEADERBOARD_PATH, MODELS_DIR, PRETRAINED_REGISTRY,
    )
    return MODELS_DIR, LEADERBOARD_PATH, PRETRAINED_REGISTRY


def _train_protocol_path() -> str:
    """Absolute path of the ASVspoof 2019 LA *train* protocol — the source of the
    attack labels used to build the unseen-attack validation split."""
    from src.model_registry import load_config
    c    = load_config()
    root = c["dataset"]["path_la2019"]
    return os.path.join(root, c["dataset"]["protocols_dir"],
                        c["dataset"]["protocols"]["train"])


def _as_float(row: Dict, key: str) -> Optional[float]:
    try:
        return float(row.get(key))
    except (TypeError, ValueError):
        return None


def _metrics_from_rows(rows: List[Dict], match: Callable[[Dict], bool]
                       ) -> Dict[str, Optional[float]]:
    """dev + eval EER/minDCF for the first model matching `match`. Operates on the
    RAW result rows (before _tag_split), where [EVAL] is still in COL_MODEL."""
    dev = next((r for r in rows if match(r)
                and "[EVAL]" not in str(r.get(COL_MODEL, ""))), {})
    ev  = next((r for r in rows if match(r)
                and "[EVAL]" in str(r.get(COL_MODEL, ""))), {})
    return {"eer_dev":  _as_float(dev, COL_EER), "mindcf_dev":  _as_float(dev, COL_MIN_DCF),
            "eer_eval": _as_float(ev,  COL_EER), "mindcf_eval": _as_float(ev,  COL_MIN_DCF),
            # Per-model operating point used by "Test an audio": the dev EER
            # threshold (where FAR = FRR on the in-domain dev split).
            "thr_dev":  _as_float(dev, COL_THRESHOLD)}


def _record(key: str, metrics: Dict) -> None:
    with _lock:
        _leaderboard[key] = metrics


def _write_leaderboard(rows: Optional[List[Dict]] = None) -> None:
    """Persist the leaderboard to leaderboard.json as
    ``{"models": {key: metrics}, "rows": [...]}``.

    `models` is merged with any existing entries (so a partial classic-only /
    CNN-only run keeps the rest). `rows` is the FULL set of per-model × split/corpus
    result rows — the web demo renders them into the SAME filterable table as local;
    it is REPLACED when a fresh (non-empty) set is given, left untouched otherwise."""
    if not _leaderboard and not rows:
        return
    _, path, _ = _registry()
    board: Dict = {"models": {}, "rows": []}
    # Read existing (migrating the legacy flat {key: metrics} format).
    for p in (path, os.path.join(os.path.dirname(path), "demo_leaderboard.json")):
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and ("models" in data or "rows" in data):
                    board = {"models": data.get("models", {}),
                             "rows": data.get("rows", [])}
                elif isinstance(data, dict):
                    board = {"models": data, "rows": []}
            except (ValueError, OSError):
                pass
            break
    with _lock:
        board["models"].update(_leaderboard)
    if rows:
        board["rows"] = rows
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(board, fh, indent=2)
    # load_leaderboard() is @st.cache_data: drop its cached copy so the Full
    # comparison page reflects the freshly written file without an app restart.
    from src.leaderboard import load_leaderboard
    load_leaderboard.clear()


def _classic_sweep(ext, feat_labels, train, primary, eval_corpora, pname, seed,
                   classic_paths=None, classic_keys=None):
    classic_paths = classic_paths or {}   # (feat, clf_name) -> .joblib path
    classic_keys  = classic_keys or {}    # (feat, clf_name) -> registry key
    rows = []
    for fk in FEATURE_ORDER:
        if _cancel.is_set():
            _stream("classic", label="Cancelled")
            break
        _step(f"Classic · {feat_labels[fk]}")
        _stream("classic", label=f"{feat_labels[fk]} — extracting…")
        x_tr, y_tr, _  = extract_feature_matrix(train, ext, fk, "train", n_workers=4, use_cache=True)
        x_pr, y_pr, ms = extract_feature_matrix(primary, ext, fk, pname, n_workers=4, use_cache=True)
        # Extract features for each chosen eval corpus and score them all.
        eval_sets = []
        for label, samples in (eval_corpora or []):
            if samples:
                x_se, y_se, _ = extract_feature_matrix(
                    samples, ext, fk, f"eval[{label}]", n_workers=4, use_cache=True)
                eval_sets.append((label, x_se, y_se))

        # Persist EVERY classifier for this front-end to its .joblib as it is
        # fitted (the sink fires inside run_classic_models after each fit).
        def _sink(name, model, _fk=fk):
            p = classic_paths.get((_fk, name))
            if p:
                try:
                    joblib.dump(model, p)
                except Exception as exc:                     # noqa: BLE001 — non-fatal
                    _stream("classic", label=f"save failed: {exc}")
        res = run_classic_models(MODEL_OPTIONS["4"], x_tr, y_tr, x_pr, y_pr,
                                 feat_labels[fk], ms, seed, eval_sets=eval_sets,
                                 model_sink=_sink)
        # Record dev/eval metrics for each classifier of this front-end.
        for cname, token in _CLF_TOKEN.items():
            key = classic_keys.get((fk, cname))
            if key:
                _record(key, _metrics_from_rows(
                    res, lambda r, t=token: t in str(r.get(COL_MODEL, ""))))
        rows.extend(_tag_split(res, pname))
        try:
            _dev = min(float(r["minDCF"]) for r in res
                       if "[EVAL]" not in str(r.get("Model", "")))
            _evs = [float(r["minDCF"]) for r in res
                    if "[EVAL]" in str(r.get("Model", ""))]
            _item = (f"{feat_labels[fk]} · dev minDCF {_dev:.3f}"
                     + (f" · eval best {min(_evs):.3f}" if _evs else ""))
        except (ValueError, KeyError):
            _item = f"{feat_labels[fk]} · done"
        _stream("classic", label=f"{feat_labels[fk]} done", inc=1, item=_item)
        _step(f"Classic · {feat_labels[fk]} done", inc=1)
    return rows


# Every deep architecture trained by the Full-comparison sweep (single source
# of truth for how many CNN slots the progress streams reserve).
_CNN_ARCHS = ["cnn", "cnn_se", "resnet", "resnext", "crnn"]


def _push_float(acc: List[float], val) -> None:
    """Append val to acc as a float, ignoring '—'/missing/unparseable entries."""
    try:
        acc.append(float(val))
    except (TypeError, ValueError):
        pass


def _aggregate_seed_rows(seed_results: List[List[Dict]], n_seeds: int) -> List[Dict]:
    """Collapse N per-seed result-row lists into ONE list of MEAN rows.

    Rows are matched across seeds by (is_eval, corpus) — i.e. the dev row and
    each eval-corpus row are averaged across seeds. EER / minDCF / accuracy and
    the timing columns become their cross-seed means, and 'EER std' / 'minDCF
    std' / 'Seeds' are added so the leaderboard can report variance. Row order
    follows the first seed (dev first, then each eval corpus)."""
    if not seed_results:
        return []
    order: List[Tuple[bool, str]] = []
    groups: Dict[Tuple[bool, str], Dict] = {}
    _avg_cols = [COL_EER, COL_MIN_DCF, COL_ACCURACY,
                 COL_TRAIN_TIME, COL_FEAT_TIME, COL_INFER_TIME, COL_THRESHOLD]
    for cres in seed_results:
        for r in cres:
            is_eval = "[EVAL]" in str(r.get(COL_MODEL, ""))
            key = (is_eval, str(r.get("Corpus", "")))
            if key not in groups:
                groups[key] = {"template": r, **{c: [] for c in _avg_cols}}
                order.append(key)
            g = groups[key]
            for c in _avg_cols:
                _push_float(g[c], r.get(c))

    out: List[Dict] = []
    for key in order:
        g = groups[key]
        row = dict(g["template"])
        if g[COL_EER]:
            row[COL_EER]   = f"{_mean(g[COL_EER]):.2f}"
            row["EER std"] = f"{_stdev(g[COL_EER]):.2f}" if len(g[COL_EER]) > 1 else "0.00"
        if g[COL_MIN_DCF]:
            row[COL_MIN_DCF]  = f"{_mean(g[COL_MIN_DCF]):.4f}"
            row["minDCF std"] = (f"{_stdev(g[COL_MIN_DCF]):.4f}"
                                 if len(g[COL_MIN_DCF]) > 1 else "0.0000")
        if g[COL_ACCURACY]:
            row[COL_ACCURACY] = f"{_mean(g[COL_ACCURACY]):.4f}"
        if g[COL_THRESHOLD]:
            row[COL_THRESHOLD] = f"{_mean(g[COL_THRESHOLD]):.4f}"
        if g[COL_TRAIN_TIME]:
            row[COL_TRAIN_TIME] = f"{_mean(g[COL_TRAIN_TIME]):.2f}"
        for c in (COL_FEAT_TIME, COL_INFER_TIME):
            if g[c]:
                row[c] = f"{_mean(g[c]):.3f}"
        row["Seeds"] = n_seeds
        out.append(row)
    return out


def _cnn_sweep(ext, base_params, train, primary, eval_corpora, seed,
               cnn_paths=None, cnn_keys=None, arch_params=None,
               seeds=None, val_samples=None):
    """Train every CNN architecture and score it on dev + each eval corpus.

    When ``seeds`` has more than one entry each architecture is trained ONCE PER
    SEED and the per-seed metrics are averaged (mean ± std) — a single run on a
    saturated dev set can't tell the architectures apart, so repeating over
    seeds is what makes the ranking trustworthy. ``val_samples`` (an unseen-attack
    holdout) drives early stopping so model selection rewards generalisation.
    The checkpoint persisted to ``cnn_paths[arch]`` is the BEST seed, picked by
    its lowest (unseen-attack) validation loss."""
    from src.models import arch_label

    cnn_paths   = cnn_paths or {}
    cnn_keys    = cnn_keys or {}
    arch_params = arch_params or {}
    seed_list   = list(seeds) if seeds else [seed]
    n_seeds     = len(seed_list)
    rows = []
    for arch in _CNN_ARCHS:
        name = arch_label(arch)
        if _cancel.is_set():
            _stream("cnn", label="Cancelled")
            break
        _step(f"CNN · {arch} ({n_seeds} seed{'s' if n_seeds > 1 else ''})")
        p = dict(base_params)
        # More DataLoader workers → faster FLAC decode each epoch (the CNN's
        # main cost, since spectrograms are re-extracted per epoch).
        p.update({"augment": True, "arch": arch,
                  "num_workers": max(int(base_params.get("num_workers", 2)), 6)})
        # Per-architecture OPTIMAL hyper-parameters (epochs / patience / batch /
        # lr …) override the shared defaults so each model trains at its best.
        p.update(arch_params.get(arch, {}))
        _ep = int(p.get("epochs", 1))

        final_ckpt   = cnn_paths.get(arch)
        per_seed_cres: List[List[Dict]] = []
        kept_tmps:     List[str] = []
        best_tmp:      Optional[str] = None
        best_val_loss  = float("inf")

        for si, sd in enumerate(seed_list):
            if _cancel.is_set():
                break
            # Fresh live chart per seed so the curves always show the current run.
            with _lock:
                _cnn_epochs.clear()
            _stream("cnn", label=f"{name} — seed {si + 1}/{n_seeds} starting…")
            pp = dict(p); pp["semilla"] = sd

            def _cb(rec, _a=name, _si=si):
                with _lock:
                    _cnn_epochs.append(dict(rec))
                _stream("cnn", label=(f"{_a} · seed {_si + 1}/{n_seeds} · epoch "
                                      f"{rec.get('epoch')}/{_ep} "
                                      f"· val={rec.get('val_loss', 0):.4f}"))

            def _cb_batch(epoch, n_batch, loss, _a=name, _si=si):
                _stream("cnn", label=(f"{_a} · seed {_si + 1}/{n_seeds} · epoch "
                                      f"{epoch}/{_ep} · batch {n_batch} "
                                      f"· loss={loss:.4f}"))

            tmp_ckpt = f"{final_ckpt}.seed{sd}.tmp" if final_ckpt else None
            _, history, cres = train_and_evaluate_cnn(
                train, primary, ext, pp, eval_sets=(eval_corpora or []),
                val_samples=val_samples, epoch_callback=_cb, batch_callback=_cb_batch,
                should_stop=_cancel.is_set, checkpoint_path=tmp_ckpt)
            if _cancel.is_set():
                break
            per_seed_cres.append(cres)
            # Seed selection: keep the checkpoint with the lowest unseen-attack
            # validation loss (the generalising one), not just the last seed.
            _vl = min((r["val_loss"] for r in history
                       if isinstance(r.get("val_loss"), (int, float))
                       and math.isfinite(r["val_loss"])), default=float("inf"))
            if tmp_ckpt:
                kept_tmps.append(tmp_ckpt)
                if _vl < best_val_loss:
                    best_val_loss = _vl
                    best_tmp      = tmp_ckpt

        # Promote the best seed's weights to the registry path; drop the rest.
        if final_ckpt and best_tmp and os.path.isfile(best_tmp):
            try:
                os.replace(best_tmp, final_ckpt)
            except OSError:
                pass
        for t in kept_tmps:
            if t != best_tmp and os.path.isfile(t):
                try:
                    os.remove(t)
                except OSError:
                    pass

        if not per_seed_cres:                 # cancelled before any seed finished
            _stream("cnn", label="Cancelled")
            break

        agg = _aggregate_seed_rows(per_seed_cres, n_seeds)
        if arch in cnn_keys and not _cancel.is_set():
            _record(cnn_keys[arch], _metrics_from_rows(agg, lambda r: True))
        rows.extend(_tag_split(agg, "dev"))
        try:
            _dev = min(float(r[COL_MIN_DCF]) for r in agg
                       if "[EVAL]" not in str(r.get(COL_MODEL, "")))
            _evs = [float(r[COL_MIN_DCF]) for r in agg
                    if "[EVAL]" in str(r.get(COL_MODEL, ""))]
            _item = (f"{name} · dev minDCF {_dev:.3f}"
                     + (f" · eval best {min(_evs):.3f}" if _evs else "")
                     + (f" · {n_seeds} seeds" if n_seeds > 1 else ""))
        except (ValueError, KeyError):
            _item = f"{name} · done"
        _stream("cnn", label=f"{name} done", inc=1, item=_item)
        _step(f"CNN · {arch} done", inc=1)
    return rows


def _raw_sweep(ext, primary, eval_corpora, pname, base_params, raw_specs):
    """Inference-only stage for the raw-waveform models (wav2vec 2.0): NEVER
    trained — just load each saved checkpoint and score the dev split + every eval
    corpus, recording its dev/eval EER & minDCF into the leaderboard. Runs after
    the CNN sweep so it has the GPU to itself. Rows are tagged like CNN rows so the
    in-page leaderboard and leaderboard.json consume them unchanged."""
    import torch

    from src.models import Wav2Vec2Classifier

    rows = []
    p = dict(base_params)
    p["num_workers"] = max(int(base_params.get("num_workers", 2)), 6)
    for key, path, name in raw_specs:
        if _cancel.is_set():
            _stream("cnn", label="Cancelled")
            break
        _step(f"{name} (evaluating)")
        _stream("cnn", label=f"{name} — loading…")
        try:
            ckpt  = torch.load(path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            model = Wav2Vec2Classifier()
            model.load_state_dict(state)
            model.eval()
        except Exception as exc:                          # noqa: BLE001 — non-fatal
            _stream("cnn", label=f"{name} load failed: {exc}", inc=1)
            _step(f"{name} failed", inc=1)
            continue

        mres = []
        _stream("cnn", label=f"{name} · scoring dev…")
        mres += evaluate_raw_on_set(model, primary, ext.sample_rate, p,
                                    corpus_label="", arch_label=name, suffix="")
        for label, samples in (eval_corpora or []):
            if _cancel.is_set() or not samples:
                continue
            _stream("cnn", label=f"{name} · scoring {label}…")
            mres += evaluate_raw_on_set(model, samples, ext.sample_rate, p,
                                        corpus_label=label, arch_label=name,
                                        suffix="[EVAL]")

        if not _cancel.is_set():
            _record(key, _metrics_from_rows(mres, lambda r: True))
        # Tag the family so the leaderboard classifies wav2vec2 as SSL, not CNN.
        _tagged = _tag_split(mres, "dev")
        for _r in _tagged:
            _r["Type"] = "SSL"
        rows.extend(_tagged)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            _best = min(float(r["minDCF"]) for r in mres
                        if "[EVAL]" not in str(r.get("Model", "")))
            _item = f"{name} · dev minDCF {_best:.3f}"
        except (ValueError, KeyError):
            _item = f"{name} · evaluated"
        _stream("cnn", label=f"{name} done", inc=1, item=_item)
        _step(f"{name} done", inc=1)
    return rows


def _run(ext, feat_labels, base_params, train, primary, eval_corpora, pname,
         classic_subset, cnn_subset, include_cnn, seed=42, export=True,
         eval_subset=0) -> Tuple[List[Dict], List[Dict]]:
    _cancel.clear()
    n_cnn = len(_CNN_ARCHS) if include_cnn else 0
    _reset_streams(len(FEATURE_ORDER), n_cnn)
    with _lock:
        _leaderboard.clear()
        _cnn_epochs.clear()   # reset so the full-comparison live view shows fresh curves
        _state.update({"total": len(FEATURE_ORDER) + n_cnn,
                       "done": 0, "label": "Preparing data…"})

    # Per-architecture optimal hyper-parameters from the YAML (epochs, patience,
    # batch size, lr …) so the Full-comparison sweep trains each model at its best.
    arch_params = {}
    cnn_seeds   = [seed]
    ms_holdout: List[str] = []
    ms_bona_frac = 0.1
    if include_cnn:
        from src.model_registry import load_config
        _cfg = load_config()
        arch_params  = dict(_cfg.get("cnn_arch_params", {}) or {})
        # Multi-seed + unseen-attack validation settings (rigour knobs): training
        # each CNN over several seeds and averaging removes single-run luck, while
        # an unseen-attack holdout gives early stopping a real generalisation
        # signal instead of the saturated, same-attack dev set.
        _ms = _cfg.get("cnn_multiseed", {}) or {}
        cnn_seeds    = list(_ms.get("seeds", [seed])) or [seed]
        ms_holdout   = list(_ms.get("holdout_attacks", []) or [])
        ms_bona_frac = float(_ms.get("bonafide_val_frac", 0.1))

    # Map the registry models to their on-disk paths so the sweeps can persist
    # them (.pth / .joblib) and record their metrics for leaderboard.json.
    classic_paths = classic_keys = cnn_paths = cnn_keys = {}
    raw_specs: List[Tuple[str, str, str]] = []
    if export:
        models_dir, _, reg = _registry()
        os.makedirs(models_dir, exist_ok=True)
        classic_paths = {(e["feat"], e["clf"]): os.path.join(models_dir, e["file"])
                         for e in reg if e["kind"] == "classic"}
        classic_keys  = {(e["feat"], e["clf"]): e["key"]
                         for e in reg if e["kind"] == "classic"}
        # Key the maps by the canonical arch key (registry "arch" field) so the
        # CNN sweep persists each architecture's checkpoint to the right file.
        cnn_paths = {e["arch"]: os.path.join(models_dir, e["file"])
                     for e in reg if e["kind"] == "cnn"}
        cnn_keys  = {e["arch"]: e["key"]
                     for e in reg if e["kind"] == "cnn"}
        # Raw-waveform models (wav2vec 2.0): evaluated-only, and only if the
        # checkpoint is actually on disk. They share the CNN progress stream.
        raw_specs = [(e["key"], os.path.join(models_dir, e["file"]), e["name"])
                     for e in reg if e["kind"] == "raw"
                     and os.path.isfile(os.path.join(models_dir, e["file"]))]
        if include_cnn and raw_specs:
            with _lock:
                _streams["cnn"]["total"] += len(raw_specs)
                _state["total"] += len(raw_specs)

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        def _sub(samples, n, s):
            return stratified_subsample(samples, n, s) if (n and n > 0) else samples

        eval_corpora = eval_corpora or []
        # Eval corpora can be enormous (2021 DF eval ≈ 600k trials), so when an
        # explicit eval_subset is given it caps EVERY eval corpus (both streams)
        # to a stratified, representative sample — the training sets keep their
        # own (typically larger / full) subset. 0 ⇒ reuse each stream's subset.
        c_ev_n = eval_subset if eval_subset else classic_subset
        n_ev_n = eval_subset if eval_subset else cnn_subset
        c_tr = _sub(train, classic_subset, seed)
        c_pr = _sub(primary, classic_subset, seed + 1)
        c_ev = [(lbl, _sub(s, c_ev_n, seed + 2)) for lbl, s in eval_corpora]
        n_tr = _sub(train, cnn_subset, seed)
        n_pr = _sub(primary, cnn_subset, seed + 1)
        n_ev = [(lbl, _sub(s, n_ev_n, seed + 2)) for lbl, s in eval_corpora]

        # Carve an UNSEEN-ATTACK validation split out of the CNN training pool: the
        # held-out attacks (e.g. A05/A06) leave training entirely and drive early
        # stopping, so model selection rewards generalisation rather than memorising
        # the train/dev-shared attacks. Empty holdout ⇒ fall back to dev (n_pr).
        cnn_val = None
        if include_cnn and ms_holdout:
            attack_ids = read_attack_ids(_train_protocol_path())
            n_tr, cnn_val = split_unseen_attacks(
                n_tr, attack_ids, ms_holdout, ms_bona_frac, seed)

        # Classic (CPU) and CNN (GPU) in parallel.
        with _cf.ThreadPoolExecutor(max_workers=2) as pool:
            f_c = pool.submit(_classic_sweep, ext, feat_labels, c_tr, c_pr, c_ev,
                              pname, seed, classic_paths, classic_keys)
            f_n = (pool.submit(_cnn_sweep, ext, base_params, n_tr, n_pr, n_ev, seed,
                               cnn_paths, cnn_keys, arch_params,
                               cnn_seeds, cnn_val)
                   if include_cnn else None)
            classic_rows = f_c.result()
            cnn_rows     = f_n.result() if f_n is not None else []

        # Raw-waveform models are evaluated AFTER the CNN training, so they get the
        # GPU to themselves (no VRAM contention on the 6 GB card). Their rows join
        # the CNN stream for the in-page leaderboard.
        if include_cnn and raw_specs and not _cancel.is_set():
            cnn_rows = cnn_rows + _raw_sweep(ext, n_pr, n_ev, pname, base_params,
                                             raw_specs)

    # Write the deployment leaderboard once the sweep finished cleanly. Persist the
    # FULL set of rows (every model × split/corpus) with their model family tagged,
    # so the web demo renders the IDENTICAL filterable table the local page shows.
    if export and not _cancel.is_set():
        persist_rows = []
        for r in classic_rows:
            d = dict(r); d["Type"] = "Classic"; persist_rows.append(d)
        for r in cnn_rows:
            d = dict(r); d["Type"] = d.get("Type") or "CNN"; persist_rows.append(d)
        _write_leaderboard(rows=persist_rows)
    return classic_rows, cnn_rows


def submit_benchmark(**kwargs) -> "_cf.Future":
    """Launch the full benchmark in the background; returns a Future."""
    return _pool.submit(_run, **kwargs)


def _run_eval_only(ext, feat_labels, eval_corpora, classic_subset,
                   base_params, seed=42) -> Tuple[List[Dict], List[Dict]]:
    """Load every saved registry model and score on eval_corpora (no training)."""
    import torch
    from src.models import arch_label as arch_label_for, model_for_arch

    _cancel.clear()
    n_cnn = len(_CNN_ARCHS)
    _reset_streams(len(FEATURE_ORDER), n_cnn)
    with _lock:
        _cnn_epochs.clear()
        _state.update({"total": len(FEATURE_ORDER) + n_cnn, "done": 0,
                       "label": "Evaluating saved models…"})

    models_dir, _, reg = _registry()
    eval_corpora = eval_corpora or []
    classic_rows: List[Dict] = []
    cnn_rows: List[Dict] = []

    # ── Classic models ─────────────────────────────────────────────────────── #
    for fk in FEATURE_ORDER:
        if _cancel.is_set():
            _stream("classic", label="Cancelled")
            break
        _stream("classic", label=f"{feat_labels[fk]} — loading saved models…")

        fitted: Dict[str, object] = {}
        for e in reg:
            if e["kind"] != "classic" or e["feat"] != fk:
                continue
            path = os.path.join(models_dir, e["file"])
            if not os.path.isfile(path):
                continue
            try:
                fitted[e["clf"]] = joblib.load(path)
            except Exception:
                pass

        if not fitted:
            _stream("classic", label=f"{feat_labels[fk]} — no saved models", inc=1)
            _step(f"Classic · {feat_labels[fk]} skipped", inc=1)
            continue

        for lbl, samples in eval_corpora:
            if not samples:
                continue
            s = (stratified_subsample(samples, classic_subset, seed + 2)
                 if classic_subset else samples)
            _stream("classic", label=f"{feat_labels[fk]} — features for {lbl}…")
            x_ev, y_ev, _ = extract_feature_matrix(
                s, ext, fk, f"eval[{lbl}]", n_workers=4, use_cache=True)
            _stream("classic", label=f"{feat_labels[fk]} — scoring on {lbl}…")
            rows = score_fitted_classic(fitted, x_ev, y_ev, feat_labels[fk],
                                        corpus_label=lbl)
            classic_rows.extend(_tag_split(rows, "dev"))

        _stream("classic", label=f"{feat_labels[fk]} done", inc=1,
                item=f"{feat_labels[fk]} evaluated")
        _step(f"Classic · {feat_labels[fk]} done", inc=1)

    # ── CNN models ─────────────────────────────────────────────────────────── #
    for e in [e for e in reg if e["kind"] == "cnn"]:
        if _cancel.is_set():
            _stream("cnn", label="Cancelled")
            break
        arch_key   = e["arch"]
        arch_label = arch_label_for(arch_key)
        path       = os.path.join(models_dir, e["file"])

        if not os.path.isfile(path):
            _stream("cnn", label=f"{arch_label} — no saved model", inc=1)
            _step(f"CNN · {arch_label} skipped", inc=1)
            continue

        _stream("cnn", label=f"{arch_label} — loading…")
        try:
            ckpt = torch.load(path, map_location=torch.device("cpu"))
            model = model_for_arch(
                ckpt.get("arch", arch_key),
                dropout=float(ckpt.get("dropout", base_params.get("dropout", 0.3))))
            model.load_state_dict(ckpt["state_dict"])
            model.to("cpu").eval()
        except Exception as exc:
            _stream("cnn", label=f"{arch_label} — load failed: {exc}", inc=1)
            _step(f"CNN · {arch_label} failed", inc=1)
            continue

        for lbl, samples in eval_corpora:
            if not samples:
                continue
            _stream("cnn", label=f"{arch_label} — evaluating on {lbl}…")
            rows = evaluate_cnn_on_set(model, samples, ext, dict(base_params),
                                       corpus_label=lbl, arch_label=arch_label)
            cnn_rows.extend(_tag_split(rows, "dev"))

        _stream("cnn", label=f"{arch_label} done", inc=1,
                item=f"{arch_label} evaluated")
        _step(f"CNN · {arch_label} done", inc=1)

    return classic_rows, cnn_rows


def submit_eval_benchmark(**kwargs) -> "_cf.Future":
    """Evaluate every saved registry model on eval_corpora; returns a Future of
    (classic_rows, cnn_rows).  No training — models must already be on disk."""
    return _pool.submit(_run_eval_only, **kwargs)
