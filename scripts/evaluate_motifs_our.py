import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_roc_curves(global_y_true, global_y_score, output_dir):
    plt.figure(figsize=(10, 8))
    
    for feature_type in sorted(global_y_true.keys()):
        if len(global_y_true[feature_type]) == 0:
            continue
        y_true = np.concatenate(global_y_true[feature_type])
        y_score = np.concatenate(global_y_score[feature_type])
        
        if len(np.unique(y_true)) < 2:
            continue
            
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f"{feature_type} (AUC = {roc_auc:.3f})")
        
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Isolated Block Strict Sampling')
    plt.legend(loc="lower right")
    
    out_path = output_dir / "roc_curves_global.pdf"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    

def plot_boxplots(global_y_true, global_y_score, output_dir):
    for feature_type in sorted(global_y_true.keys()):
        if len(global_y_true[feature_type]) == 0:
            continue
        y_true = np.concatenate(global_y_true[feature_type])
        y_score = np.concatenate(global_y_score[feature_type])
        
        if len(np.unique(y_true)) < 2:
            continue
            
        plt.figure(figsize=(8, 6))
        pos_scores = y_score[y_true == 1]
        neg_scores = y_score[y_true == 0]
        
        data = [neg_scores]
        labels = ['Background']
        if len(pos_scores) > 0:
            data.append(pos_scores)
            labels.append(feature_type)
            
        plt.boxplot(data, labels=labels)
        
        plt.title(f'Block Score Distribution: {feature_type} vs Background')
        plt.ylabel('Block Score')
        
        out_path = output_dir / f"boxplot_{feature_type}.pdf"
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()


def get_optimal_metrics(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]
    
    y_pred = (y_score >= best_threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    return roc_auc, accuracy, precision, recall, best_threshold


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/evaluate_motifs_our.py <run_dir>")
        sys.exit(1)
        
    run_dir = Path(sys.argv[1])
    
    if not run_dir.is_dir():
        print(f"Error: Not a valid directory: {run_dir}")
        sys.exit(1)
        
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    regions_path = Path(manifest["regions_path"])
    if not regions_path.exists():
        repo_root = Path(__file__).resolve().parents[1]
        regions_path = repo_root / regions_path
        
    regions_df = pd.read_parquet(regions_path)
    
    per_span_path = run_dir / "block_scores" / "per_span.parquet"
    if not per_span_path.exists():
        raise FileNotFoundError(f"per_span.parquet not found at {per_span_path}")
        
    per_span = pd.read_parquet(per_span_path)
    
    output_dir = run_dir / "motif_evaluation_our"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    region_groups = per_span.groupby(["region_id", "record_id", "region_start", "region_end", "region_length"])
    
    global_y_true = {}
    global_y_score = {}
    
    # Class imbalance parameter: number of sampled background windows per motif
    N_BACKGROUND_WINDOWS = 5

    np.random.seed(42) # For reproducible sampling

    print(f"Evaluating regions using isolated block strict sampling (N={N_BACKGROUND_WINDOWS})...")
    for (region_id, record_id, region_start, region_end, region_length), group in tqdm(region_groups):
        block_size = group['block_size'].iloc[0]
        
        y_score = np.full(region_length, np.nan)
        min_dist = np.full(region_length, np.inf)
        span_starts_at_pos = np.full(region_length, -1)
        
        span_centers = group['span_start'].values + block_size // 2
        tile_centers = group['tile_start'].values + group['tile_length'].values / 2
        
        dists = np.abs(span_centers - tile_centers)
        region_indices = span_centers - region_start
        
        for idx, pos in zip(group.index, region_indices):
            if 0 <= pos < region_length:
                dist = dists[group.index.get_loc(idx)]
                if dist < min_dist[pos]:
                    min_dist[pos] = dist
                    y_score[pos] = group.loc[idx, 'block_score']
                    span_starts_at_pos[pos] = group.loc[idx, 'span_start'] - region_start
        
        map_annots = regions_df[
            (regions_df['record_id'] == record_id) & 
            (regions_df['label'] != 'CompositePromoter') &
            (regions_df['start'] < region_end) & 
            (regions_df['end'] > region_start)
        ]
        
        is_annotated = np.zeros(region_length, dtype=bool)
        
        for _, annot in map_annots.iterrows():
            ann_start = max(0, int(annot['start']) - region_start)
            ann_end = min(region_length, int(annot['end']) - region_start)
            if ann_start < ann_end:
                is_annotated[ann_start:ann_end] = True
                
        for _, annot in map_annots.iterrows():
            ann_start = max(0, int(annot['start']) - region_start)
            ann_end = min(region_length, int(annot['end']) - region_start)
            L = ann_end - ann_start
            
            if L < block_size:
                continue # Motif is smaller than block size, cannot strictly enclose a block
                
            ftype = annot['feature_type']
            if ftype not in ["Core", "TTF1", "CTCF"]:
                continue
                
            if ftype not in global_y_true:
                global_y_true[ftype] = []
                global_y_score[ftype] = []
                
            # POSITIVE CLASS: strict block enclosure
            pos_mask = (span_starts_at_pos >= ann_start) & (span_starts_at_pos + block_size <= ann_end) & (~np.isnan(y_score))
            
            y_s_pos = y_score[pos_mask]
            y_t_pos = np.ones(len(y_s_pos), dtype=int)
            
            if len(y_s_pos) > 0:
                global_y_score[ftype].append(y_s_pos)
                global_y_true[ftype].append(y_t_pos)
            
            # NEGATIVE CLASS: sample N unannotated windows of length L
            valid_bg_starts = []
            for bg_s in range(region_length - L + 1):
                if not np.any(is_annotated[bg_s:bg_s+L]):
                    valid_bg_starts.append(bg_s)
                    
            if len(valid_bg_starts) > 0:
                n_samples = min(N_BACKGROUND_WINDOWS, len(valid_bg_starts))
                sampled_bg_starts = np.random.choice(valid_bg_starts, size=n_samples, replace=False)
                
                for bg_start in sampled_bg_starts:
                    bg_end = bg_start + L
                    bg_mask = (span_starts_at_pos >= bg_start) & (span_starts_at_pos + block_size <= bg_end) & (~np.isnan(y_score))
                    y_s_neg = y_score[bg_mask]
                    y_t_neg = np.zeros(len(y_s_neg), dtype=int)
                    
                    if len(y_s_neg) > 0:
                        global_y_score[ftype].append(y_s_neg)
                        global_y_true[ftype].append(y_t_neg)

    print("Computing global metrics...")
    global_metrics = []
    for ftype in sorted(global_y_true.keys()):
        if len(global_y_true[ftype]) == 0:
            continue
            
        y_t = np.concatenate(global_y_true[ftype])
        y_s = np.concatenate(global_y_score[ftype])
        
        roc_auc, acc, prec, rec, best_thresh = get_optimal_metrics(y_t, y_s)
        
        global_metrics.append({
            "feature_type": ftype,
            "auroc": roc_auc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "best_threshold": best_thresh,
            "n_positive": np.sum(y_t == 1),
            "n_negative": np.sum(y_t == 0)
        })
        
    global_metrics_df = pd.DataFrame(global_metrics)
    global_metrics_df.to_csv(output_dir / "metrics_global.csv", index=False)
    
    with open(output_dir / "metrics_global.json", "w") as f:
        json.dict_list = global_metrics_df.to_dict('records')
        json.dump(json.dict_list, f, indent=2)

    print("Generating plots...")
    plot_roc_curves(global_y_true, global_y_score, output_dir)
    plot_boxplots(global_y_true, global_y_score, output_dir)
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
