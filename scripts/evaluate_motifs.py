import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score
import sys
from pathlib import Path

# Add dependency_map to path
sys.path.insert(0, str(Path('external/dependency_map/src').resolve()))
from dependency_map import DependencyMap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_roc_curves(global_y_true, global_y_score, output_dir):
    plt.figure(figsize=(10, 8))
    
    for feature_type in sorted(global_y_true.keys()):
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
    plt.title('Receiver Operating Characteristic (ROC) - Single Nucleotide')
    plt.legend(loc="lower right")
    
    out_path = output_dir / "roc_curves_global.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    plt.close()
    

def plot_boxplots(global_y_true, global_y_score, output_dir):
    for feature_type in sorted(global_y_true.keys()):
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
        
        out_path = output_dir / f"boxplot_{feature_type}.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close()


def plot_discovered_motifs(novel_df, per_span, output_dir, window_size=200):
    if novel_df.empty:
        return
        
    plots_dir = output_dir / "discovered_motifs_plots"
    plots_dir.mkdir(exist_ok=True)
    
    for _, row in novel_df.iterrows():
        record_id = row['record_id']
        region_id = row['region_id']
        motif_start = row['start_position']
        motif_end = row['end_position']
        ftype = row['feature_type']
        
        motif_center = (motif_start + motif_end) // 2
        window_start = motif_center - (window_size // 2)
        window_end = motif_center + (window_size // 2)
        
        # Filter per_span
        df = per_span[(per_span['record_id'] == record_id) & (per_span['region_id'] == region_id)].copy()
        if df.empty:
            continue
            
        block_size = df['block_size'].iloc[0]
        df['span_center'] = df['span_start'] + block_size / 2
        
        window_df = df[(df['span_center'] >= window_start) & (df['span_center'] <= window_end)]
        window_df = window_df.sort_values('span_center')
        
        plt.figure(figsize=(10, 6))
        plt.axvspan(motif_start, motif_end, color='red', alpha=0.2, label=f'Discovered {ftype} Motif')
        plt.plot(window_df['span_center'], window_df['block_score'], color='blue', linewidth=2, label='Dependency Score')
        plt.scatter(window_df['span_center'], window_df['block_score'], color='blue', s=15)
        
        plt.xlabel('Genomic Position')
        plt.ylabel('Block Score')
        plt.title(f'Dependency Map around Discovered Motif ({ftype})\n{record_id}:{motif_start}-{motif_end}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Create safe filename
        safe_rec = str(record_id).replace('/', '_').replace(' ', '_')
        out_path = plots_dir / f"{ftype}_{safe_rec}_{motif_start}.png"
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        # --- True 2D Dependency Map ---
        try:
            # Locate map_index.parquet
            run_dir = output_dir.parent
            map_index_path = run_dir / "dependency_maps" / "map_index.parquet"
            if map_index_path.exists():
                map_idx_df = pd.read_parquet(map_index_path)
                
                # Find the map that covers the motif center
                matches = map_idx_df[(map_idx_df['region_id'] == region_id) & 
                                     (map_idx_df['start'] <= motif_center) & 
                                     (map_idx_df['end'] >= motif_center)]
                
                if not matches.empty:
                    match = matches.iloc[0]
                    map_path = run_dir / match['map_path']
                    
                    data = np.load(map_path)
                    map_2d = data['dependency_map']
                    recon = data['reconstruction']
                    seq_str = ''.join(data['sequence'].astype(str))
                    
                    map_start = match['start']
                    map_end = match['end']
                    
                    # Calculate window bounds relative to the map
                    win_start_genomic = max(map_start, motif_center - window_size // 2)
                    win_end_genomic = min(map_end, motif_center + window_size // 2)
                    
                    start_idx = win_start_genomic - map_start
                    end_idx = win_end_genomic - map_start
                    
                    sub_map = map_2d[start_idx:end_idx, start_idx:end_idx]
                    sub_recon = recon[start_idx:end_idx]
                    sub_seq = seq_str[start_idx:end_idx]
                    
                    dm = DependencyMap(sub_seq, sub_map, sub_recon)
                    fig = dm.plot()
                    
                    # Highlight motif
                    # Relative position in the window
                    rel_motif_start = motif_start - win_start_genomic
                    rel_motif_end = motif_end - win_start_genomic
                    
                    fig.add_shape(
                        type="rect",
                        x0=rel_motif_start, y0=rel_motif_start, 
                        x1=rel_motif_end, y1=rel_motif_end,
                        line=dict(color="red", width=1),
                        fillcolor="rgba(0,0,0,0)"
                    )
                    
                    out_html = plots_dir / f"{ftype}_{safe_rec}_{motif_start}_true2d.html"
                    out_png_2d = plots_dir / f"{ftype}_{safe_rec}_{motif_start}_true2d.png"
                    
                    fig.write_html(str(out_html))
                    fig.write_image(str(out_png_2d), scale=3)
        except Exception as e:
            print(f"Warning: Failed to generate true 2D dependency map for {motif_start}: {e}")

def get_optimal_metrics(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    # Youden's J statistic
    J = tpr - fpr
    best_idx = np.argmax(J)
    best_threshold = thresholds[best_idx]
    
    y_pred = (y_score >= best_threshold).astype(int)
    
    # True positives, False Positives, etc.
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    return roc_auc, accuracy, precision, recall, best_threshold


def discover_novel_motifs(full_nuc_df, threshold, block_size):
    discovered = []
    
    if np.isnan(threshold) or threshold == np.inf:
        return pd.DataFrame(discovered)
        
    # Filter for unannotated positives exceeding the strict threshold
    fp_df = full_nuc_df[(~full_nuc_df['is_annotated_discovery']) & (full_nuc_df['block_score'] >= threshold)].copy()
    if fp_df.empty:
        return pd.DataFrame(discovered)
        
    fp_df = fp_df.sort_values(['record_id', 'region_id', 'position'])
    
    length_bounds = {
        'Core': (55, 168),
        'CTCF': (20, 20),
        'TTF1': (18, 18)
    }
    
    # Group to find contiguous spans
    for (rec_id, reg_id), group in fp_df.groupby(['record_id', 'region_id']):
        group['block_id'] = (group['position'].diff() > 1).cumsum()
        
        for block_id, span in group.groupby('block_id'):
            seq_length = len(span) + block_size - 1
            
            for ftype, (min_len, max_len) in length_bounds.items():
                if min_len <= seq_length <= max_len:
                    start_pos = span['position'].min() - block_size // 2
                    end_pos = start_pos + seq_length
                    
                    discovered.append({
                        'feature_type': ftype,
                        'record_id': rec_id,
                        'region_id': reg_id,
                        'start_position': start_pos,
                        'end_position': end_pos,
                        'genomic_length': seq_length,
                        'center_count': len(span),
                        'mean_score': span['block_score'].mean(),
                        'max_score': span['block_score'].max()
                    })
                
    return pd.DataFrame(discovered)

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/evaluate_motifs.py <run_dir>")
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
        # Fallback to repo root if path is relative
        repo_root = Path(__file__).resolve().parents[1]
        regions_path = repo_root / regions_path
        
    regions_df = pd.read_parquet(regions_path)
    
    per_span_path = run_dir / "block_scores" / "per_span.parquet"
    if not per_span_path.exists():
        raise FileNotFoundError(f"per_span.parquet not found at {per_span_path}")
        
    per_span = pd.read_parquet(per_span_path)
    
    output_dir = run_dir / "motif_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group spans by region_id to assemble the full promoter
    region_groups = per_span.groupby(["region_id", "record_id", "region_start", "region_end", "region_length"])
    
    all_nucleotides = []
    global_y_true = {}
    global_y_score = {}
    
    file_metrics = []
    
    # Get all unique feature types to evaluate every motif across all regions
    all_feature_types = ["Core", "TTF1", "CTCF"]

    print("Evaluating regions...")
    for (region_id, record_id, region_start, region_end, region_length), group in tqdm(region_groups):
        block_size = group['block_size'].iloc[0]
        
        y_score = np.full(region_length, np.nan)
        min_dist = np.full(region_length, np.inf)
        
        # Determine center of each span and its tile
        span_centers = group['span_start'].values + block_size // 2
        tile_centers = group['tile_start'].values + group['tile_length'].values / 2
        
        # Calculate distances to tile centers
        dists = np.abs(span_centers - tile_centers)
        
        # Map to region coordinates
        region_indices = span_centers - region_start
        
        # Iterate over all spans to populate y_score using minimum distance to tile center
        for idx, pos in zip(group.index, region_indices):
            # pos could be out of bounds if span center is outside region (shouldn't happen but just in case)
            if 0 <= pos < region_length:
                dist = dists[group.index.get_loc(idx)]
                if dist < min_dist[pos]:
                    min_dist[pos] = dist
                    y_score[pos] = group.loc[idx, 'block_score']
        
        # Get annotations for this record overlapping the region
        map_annots = regions_df[
            (regions_df['record_id'] == record_id) & 
            (regions_df['label'] != 'CompositePromoter') &
            (regions_df['start'] < region_end) & 
            (regions_df['end'] > region_start)
        ]
        
        is_annotated_strict = np.zeros(region_length, dtype=bool)
        is_annotated_discovery = np.zeros(region_length, dtype=bool)
        feature_masks = {}
        
        for _, annot in map_annots.iterrows():
            ann_start = max(0, annot['start'] - region_start)
            ann_end = min(region_length, annot['end'] - region_start)
            if ann_start < ann_end:
                ftype = annot['feature_type']
                is_annotated_strict[ann_start:ann_end] = True
                if ftype not in ['EnhancerRepeats', 'Upstream']:
                    is_annotated_discovery[ann_start:ann_end] = True
                if ftype not in feature_masks:
                    feature_masks[ftype] = np.zeros(region_length, dtype=bool)
                feature_masks[ftype][ann_start:ann_end] = True
                
        # Collect nucleotide data for saving
        abs_positions = np.arange(region_start, region_end)
        nuc_df = pd.DataFrame({
            "region_id": region_id,
            "record_id": record_id,
            "position": abs_positions,
            "block_score": y_score,
            "is_annotated_strict": is_annotated_strict,
            "is_annotated_discovery": is_annotated_discovery
        })
        for ftype in all_feature_types:
            nuc_df[f"feature_{ftype}"] = feature_masks.get(ftype, np.zeros(region_length, dtype=bool))
        all_nucleotides.append(nuc_df)
        
        # Evaluate for each feature type globally across all regions
        for ftype in all_feature_types:
            if ftype not in global_y_true:
                global_y_true[ftype] = []
                global_y_score[ftype] = []
                
            y_true_M = feature_masks.get(ftype, np.zeros(region_length, dtype=bool))
            valid_mask = ((y_true_M == True) | (~is_annotated_strict)) & (~np.isnan(y_score))
            
            if np.sum(valid_mask) == 0:
                continue
                
            y_t = y_true_M[valid_mask].astype(int)
            y_s = y_score[valid_mask]
            
            global_y_true[ftype].append(y_t)
            global_y_score[ftype].append(y_s)
            
            roc_auc, acc, prec, rec, best_thresh = get_optimal_metrics(y_t, y_s)
            
            file_metrics.append({
                "region_id": region_id,
                "record_id": record_id,
                "feature_type": ftype,
                "auroc": roc_auc,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "best_threshold": best_thresh,
                "n_positive": np.sum(y_t == 1),
                "n_negative": np.sum(y_t == 0)
            })

    print("Aggregating single nucleotide dataframe...")
    if all_nucleotides:
        full_nuc_df = pd.concat(all_nucleotides, ignore_index=True)
        full_nuc_df.to_parquet(output_dir / "single_nucleotide_scores.parquet", index=False)
        
    print("Computing global metrics...")
    global_metrics = []
    best_thresholds = {}
    for ftype in sorted(global_y_true.keys()):
        y_t = np.concatenate(global_y_true[ftype])
        y_s = np.concatenate(global_y_score[ftype])
        
        roc_auc, acc, prec, rec, best_thresh = get_optimal_metrics(y_t, y_s)
        best_thresholds[ftype] = best_thresh
        
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
    
    file_metrics_df = pd.DataFrame(file_metrics)
    file_metrics_df.to_csv(output_dir / "metrics_per_file.csv", index=False)
    
    with open(output_dir / "metrics_global.json", "w") as f:
        json.dict_list = global_metrics_df.to_dict('records')
        json.dump(json.dict_list, f, indent=2)

    print("Generating plots...")
    plot_roc_curves(global_y_true, global_y_score, output_dir)
    plot_boxplots(global_y_true, global_y_score, output_dir)
    
    print("Discovering novel motifs...")
    if all_nucleotides:
        block_size = per_span['block_size'].iloc[0]
        
        bg_scores = full_nuc_df.loc[~full_nuc_df['is_annotated_discovery'], 'block_score'].dropna()
        if len(bg_scores) > 0:
            p95_thresh = np.percentile(bg_scores, 95)
        else:
            p95_thresh = np.inf
            
        print(f"Using 95th percentile threshold for discovery: {p95_thresh:.4f}")
        
        novel_df = discover_novel_motifs(full_nuc_df, p95_thresh, block_size)
        novel_df.to_csv(output_dir / "discovered_novel_motifs.csv", index=False)
        print(f"Found {len(novel_df)} novel motif spans.")
        if len(novel_df) > 0:
            print("Plotting discovered motifs...")
            plot_discovered_motifs(novel_df, per_span, output_dir)
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
