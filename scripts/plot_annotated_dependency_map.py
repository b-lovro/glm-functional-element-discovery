import argparse
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Add dependency_map to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / 'external/dependency_map/src'))
from dependency_map import DependencyMap

def get_base_color(ftype):
    """Assigns specific colors to primary motif features."""
    colors = {
        'Core': 'rgba(255, 0, 0, 1.0)',
        'CTCF': 'rgba(0, 0, 255, 1.0)',
        'TTF1': 'rgba(0, 255, 0, 1.0)',
    }
    for key, color in colors.items():
        if key in ftype:
            return color
    return None

def main():
    parser = argparse.ArgumentParser(description="Plot an interactive 2D annotated dependency map for a specific evaluated region.")
    parser.add_argument("run_dir", type=str, help="Path to the model run directory (e.g. outputs/runs/pretraining_...).")
    parser.add_argument("region_id", type=str, help="The exact region ID to plot (e.g. 'Mammalia...-morph7:9').")
    parser.add_argument("--tile-index", type=int, default=0, help="The tile index to plot if the region was chunked. Default is 0.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for the plots. Defaults to <run_dir>/motif_evaluation.")
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
        
    map_index_path = run_dir / "dependency_maps" / "map_index.parquet"
    if not map_index_path.exists():
        print(f"Error: Map index not found at {map_index_path}. Were dependency maps generated?")
        sys.exit(1)
        
    map_index = pd.read_parquet(map_index_path)
    
    # Filter for the requested region and tile
    matches = map_index[(map_index['region_id'] == args.region_id) & (map_index['tile_index'] == args.tile_index)]
    if matches.empty:
        print(f"Error: Could not find region_id '{args.region_id}' with tile_index '{args.tile_index}' in map_index.")
        print(f"Available tiles for this region: {map_index[map_index['region_id'] == args.region_id]['tile_index'].tolist()}")
        sys.exit(1)
        
    match = matches.iloc[0]
    map_path = run_dir / match['map_path']
    record_id = match['record_id']
    tile_start = match['start']
    tile_end = match['end']
    
    region_start = match['region_start']
    region_end = match['region_end']
    region_length = match['region_length']
    
    if not map_path.exists():
        print(f"Error: Map file not found at {map_path}")
        sys.exit(1)
        
    print(f"Loading map for {args.region_id} (tile {args.tile_index}) spanning {tile_start}-{tile_end}...")
    data = np.load(map_path)
    map_2d = data['dependency_map']
    recon = data['reconstruction']
    seq_str = ''.join(data['sequence'].astype(str))
    
    # Generate the base DependencyMap
    dm = DependencyMap(seq_str, map_2d, recon)
    fig = dm.plot()
    
    # Format axes to show nucleotide characters along the borders
    fig.update_xaxes(
        visible=True,
        tickmode='array',
        tickvals=np.arange(len(seq_str)),
        ticktext=list(seq_str),
        tickangle=0,
        tickfont=dict(size=8),
        showgrid=False,
        zeroline=False,
        scaleanchor="y",
        constrain="domain",
        row=1, col=1
    )
    fig.update_yaxes(
        visible=True,
        tickmode='array',
        tickvals=np.arange(len(seq_str)),
        ticktext=list(seq_str),
        tickfont=dict(size=8),
        showgrid=False,
        zeroline=False,
        scaleanchor="x",
        constrain="domain",
        autorange="reversed",
        row=1, col=1
    )
    
    # Remove all sequence logos except the top one
    if hasattr(fig.layout, 'images') and fig.layout.images:
        fig.layout.images = [img for img in fig.layout.images if img.y == 1.0 and img.yanchor == 'bottom']
    

    
    # Add Overview Track / Slider under the map
    # Shrink the main yaxis domain to make room at the bottom
    fig.update_layout(
        yaxis=dict(domain=[0.10, 1.0]),
        legend=dict(x=1.05, y=1.0, xanchor='left', yanchor='top')
    )
    
    # Hide the heatmap colorbar
    fig.update_traces(showscale=False, selector=dict(type='heatmap'))
    
    # Load manifest and annotations
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)
        
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    records_path = Path(manifest.get("records_path", "data/prepared/ribosome/records.parquet"))
    if not records_path.exists():
        records_path = repo_root / records_path
    
    if records_path.exists():
        records = pd.read_parquet(records_path)
        fasta_length = records[records['record_id'] == record_id]['sequence_length'].iloc[0]
    else:
        fasta_length = region_end # Fallback if not found
    
    fig.update_layout(
        xaxis2=dict(
            domain=[0.25, 0.75], 
            anchor='y2',
            range=[0, fasta_length],
            title=dict(text="Genomic Sequence Coordinate (Fasta File)", font=dict(size=12)),
            tickvals=[0, tile_start, tile_end, fasta_length],
            ticktext=['0', str(tile_start), str(tile_end), str(fasta_length)],
            tickangle=45,
            tickfont=dict(size=11)
        ),
        yaxis2=dict(
            domain=[0.0, 0.03], 
            anchor='x2',
            showticklabels=False,
            range=[0, 1]
        )
    )
    
    # Background for the whole region
    fig.add_shape(
        type='rect',
        x0=0, y0=0, x1=fasta_length, y1=1,
        fillcolor='lightgray', line_width=0,
        xref='x2', yref='y2'
    )
    
    # Highlight for the current tile
    fig.add_shape(
        type='rect',
        x0=tile_start, y0=0, x1=tile_end, y1=1,
        fillcolor='rgba(0, 0, 255, 0.5)', line_width=2, line_color='black',
        xref='x2', yref='y2'
    )
    
    # Setup output directory
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "motif_evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_region = args.region_id.replace('/', '_').replace(' ', '_').replace(':', '_')

    regions_path = Path(manifest["regions_path"])
    if not regions_path.exists():
        regions_path = repo_root / regions_path
        
    if regions_path.exists():
        print(f"Loading annotations from {regions_path}...")
        regions = pd.read_parquet(regions_path)
        
        map_annots = regions[
            (regions['record_id'] == record_id) & 
            (regions['label'] != 'CompositePromoter') &
            (regions['feature_type'] != 'EnhancerRepeats') &
            (regions['start'] < tile_end) & 
            (regions['end'] > tile_start)
        ]
        
        print(f"Found {len(map_annots)} annotations in this tile.")
        
        unique_ftypes = map_annots['feature_type'].unique()
        plotly_colors = [
            'rgba(31, 119, 180, 1.0)', 'rgba(255, 127, 14, 1.0)', 'rgba(44, 160, 44, 1.0)',
            'rgba(148, 103, 189, 1.0)', 'rgba(140, 86, 75, 1.0)', 'rgba(227, 119, 194, 1.0)', 
            'rgba(127, 127, 127, 1.0)', 'rgba(188, 189, 34, 1.0)', 'rgba(23, 190, 207, 1.0)'
        ]
        
        color_map = {}
        c_idx = 0
        for ftype in unique_ftypes:
            base_c = get_base_color(ftype)
            if base_c:
                color_map[ftype] = base_c
            else:
                color_map[ftype] = plotly_colors[c_idx % len(plotly_colors)]
                c_idx += 1
                
        # Add Dynamic Legend
        for name, color in color_map.items():
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=12, color=color, symbol='square'),
                name=name, showlegend=True,
                hoverinfo='none'
            ))
        
        for _, annot in map_annots.iterrows():
            ann_start = max(0, annot['start'] - tile_start)
            ann_end = min(tile_end - tile_start, annot['end'] - tile_start)
            
            # Draw highlight rectangle
            ftype = annot['feature_type']
            color = color_map[ftype]
            
            fig.add_shape(
                type="rect",
                x0=ann_start, y0=ann_start, 
                x1=ann_end, y1=ann_end,
                line=dict(color=color, width=1),
                fillcolor='rgba(0,0,0,0)'
            )
            
            # Plot specific zoomed-in dependency map for Core, CTCF, TTF1
            if ftype in ['Core', 'CTCF', 'TTF1'] and ann_end > ann_start:
                sub_map = map_2d[ann_start:ann_end, ann_start:ann_end]
                sub_recon = recon[ann_start:ann_end] if recon is not None else None
                sub_seq = seq_str[ann_start:ann_end]
                
                dm_sub = DependencyMap(sub_seq, sub_map, sub_recon)
                fig_sub = dm_sub.plot()
                
                # Format axes for sub-plot
                fig_sub.update_xaxes(
                    visible=True, tickmode='array', tickvals=np.arange(len(sub_seq)), 
                    ticktext=list(sub_seq), tickangle=0, tickfont=dict(size=10),
                    showgrid=False, zeroline=False, scaleanchor="y", constrain="domain", row=1, col=1
                )
                fig_sub.update_yaxes(
                    visible=True, tickmode='array', tickvals=np.arange(len(sub_seq)),
                    ticktext=list(sub_seq), tickfont=dict(size=10),
                    showgrid=False, zeroline=False, scaleanchor="x", constrain="domain", autorange="reversed", row=1, col=1
                )
                if hasattr(fig_sub.layout, 'images') and fig_sub.layout.images:
                    fig_sub.layout.images = [img for img in fig_sub.layout.images if img.y == 1.0 and img.yanchor == 'bottom']
                
                # Hide the heatmap colorbar
                fig_sub.update_traces(showscale=False, selector=dict(type='heatmap'))
                
                sub_out_html = out_dir / f"{safe_region}_tile{args.tile_index}_{ftype}_{annot['start']}_{annot['end']}.html"
                sub_out_pdf = out_dir / f"{safe_region}_tile{args.tile_index}_{ftype}_{annot['start']}_{annot['end']}.pdf"
                fig_sub.write_html(str(sub_out_html))
                fig_sub.write_image(str(sub_out_pdf))
    else:
        print(f"Warning: Regions file not found at {regions_path}. No annotations will be highlighted.")

    # Save full tile outputs
    out_html = out_dir / f"{safe_region}_tile{args.tile_index}_annotated.html"
    out_pdf = out_dir / f"{safe_region}_tile{args.tile_index}_annotated.pdf"
    
    print(f"Saving interactive HTML to {out_html}...")
    fig.write_html(str(out_html))
    print(f"Saving static PDF to {out_pdf}...")
    fig.write_image(str(out_pdf))
    print("Done!")

if __name__ == "__main__":
    main()
