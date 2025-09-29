
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path

# Add project root to sys.path to allow importing moellava
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# set plt font: 'Monaco'
plt.rcParams['font.family'] = 'Monaco'

from moellava.model.multimodal_model.qwen2_vl_moe import EvalMoEQwen2VLForConditionalGeneration, AdaptiveGroupingMoE, CombinedLayer

def analyze_expert_grouping(model_path, output_dir):
    """
    Loads a trained MoEQwen2VLForConditionalGeneration model and analyzes
    the expert grouping in its AdaptiveGroupingMoE layers.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}...")
    # Load model on CPU to avoid potential VRAM issues during analysis.
    # The analysis itself is not compute-intensive.
    model = EvalMoEQwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16, # Use bfloat16 for consistency, though float32 is fine for CPU
        device_map="cpu"
    )
    model.eval()

    print("Analyzing expert grouping for each AdaptiveGroupingMoE layer...")

    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            mlp_module = layer.mlp
            
            # The MoE layer can be nested inside a CombinedLayer
            moe_layer = None
            if isinstance(mlp_module, AdaptiveGroupingMoE):
                moe_layer = mlp_module
            elif isinstance(mlp_module, CombinedLayer) and isinstance(mlp_module.moe, AdaptiveGroupingMoE):
                moe_layer = mlp_module.moe
            
            if moe_layer is not None:
                print(f"\n--- Analyzing Layer {i} (AdaptiveGroupingMoE) ---")
                
                # Get group assignments using the model's internal method
                hard_assignment, group_sizes, _ = moe_layer._get_group_assignment()
                
                hard_assignment = hard_assignment.cpu().float().numpy() # [max_groups, num_experts]
                group_sizes = group_sizes.cpu().float().numpy().astype(int)

                active_groups_mask = group_sizes > 0
                num_active_groups = active_groups_mask.sum()
                active_group_indices = np.where(active_groups_mask)[0]

                print(f"Number of active groups: {num_active_groups} / {moe_layer.max_groups}")
                
                if num_active_groups > 0:
                    print(f"Active group sizes: {group_sizes[active_groups_mask]}")
                    
                    # Determine which group each expert belongs to
                    print(f'{i} hard_assignment: {hard_assignment}')
                    expert_to_group_map = np.argmax(hard_assignment, axis=0)
                    print(f'{i} expert_to_group_map: {expert_to_group_map}')
                    
                    # Print expert assignments for each active group
                    for group_idx in active_group_indices:
                        experts_in_group = np.where(expert_to_group_map == group_idx)[0]
                        print(f"  Group {group_idx} ({group_sizes[group_idx]} experts): {experts_in_group.tolist()}")

                    # --- New Visualization 1: Bar Chart for Group Sizes ---
                    plt.figure(figsize=(5, 3), dpi=600)
                    active_group_sizes = group_sizes[active_groups_mask]
                    sns.barplot(x=active_group_indices.astype(str), y=active_group_sizes, palette="viridis")
                    # plt.title(f"Layer {i}: Group Size Distribution ({num_active_groups} Active Groups)")
                    plt.xlabel("Group Index")
                    plt.ylabel("Number of Experts")
                    plt.xticks(rotation=90)
                    # Make xticks smaller if there are many groups
                    if num_active_groups > 20:
                        plt.tick_params(axis='x', which='major', labelsize=8, bottom=False, labelbottom=False)
                    plt.tight_layout()
                    plot_path_sizes = os.path.join(output_dir, f"layer_{i}_group_sizes.png")
                    plt.savefig(plot_path_sizes, bbox_inches='tight')
                    plt.close()
                    print(f"Saved group size bar chart to {plot_path_sizes}")

                    # --- New Visualization 2: Clustered Heatmap ---
                    # Reorder experts on the x-axis so that experts in the same group are adjacent
                    sorted_expert_indices = np.argsort(expert_to_group_map)
                    reordered_assignment = hard_assignment[:, sorted_expert_indices]
                    reordered_expert_to_group = expert_to_group_map[sorted_expert_indices]

                    # Filter for active groups only for the y-axis
                    active_reordered_assignment = reordered_assignment[active_groups_mask, :]
                    
                    plt.figure(figsize=(20, 8))
                    ax = sns.heatmap(
                        active_reordered_assignment,
                        cmap="viridis",
                        cbar=False,
                        yticklabels=active_group_indices,
                        xticklabels=sorted_expert_indices
                    )
                    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

                    # Make x-axis labels less crowded if there are many experts
                    if moe_layer.num_experts > 100:
                        tick_skip = max(1, moe_layer.num_experts // 100) # Show at most 100 ticks
                        ax.set_xticks(ax.get_xticks()[::tick_skip])
                        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=8)
                    else:
                         ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')

                    plt.title(f"Layer {i}: Clustered Expert-to-Group Assignment")
                    plt.xlabel("Expert Index (Reordered by Group)")
                    plt.ylabel("Group Index")

                    # Add vertical lines to visually separate the groups
                    group_boundaries = np.where(np.diff(reordered_expert_to_group) != 0)[0] + 1
                    for boundary in group_boundaries:
                        plt.axvline(x=boundary, color='white', linewidth=0.75, linestyle='--')
                    
                    plot_path_clustered = os.path.join(output_dir, f"layer_{i}_clustered_assignment.png")
                    plt.savefig(plot_path_clustered, bbox_inches='tight')
                    plt.close()
                    print(f"Saved clustered assignment heatmap to {plot_path_clustered}")

                else:
                    print("No active groups found for this layer.")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze expert grouping in a trained MoEQwen2VLForConditionalGeneration model with AdaptiveGroupingMoE layers."
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        required=True, 
        help="Path to the directory containing the trained model checkpoint (e.g., the output of a training run)."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./group_analysis_results", 
        help="Directory to save analysis results and visualizations."
    )
    args = parser.parse_args()

    analyze_expert_grouping(args.model_path, args.output_dir)

if __name__ == "__main__":
    main()
