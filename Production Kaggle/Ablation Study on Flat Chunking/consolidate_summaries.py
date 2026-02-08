
import json
import os
import pandas as pd
from collections import OrderedDict

# Paths
BASE_DIR = "/Users/mrarnav69/Documents/TreeSum/Production Kaggle/Ablation Study on Flat Chunking"
INDEX_FILE_SET_A = os.path.join(BASE_DIR, "shared_sample_indices.json")

PATHS = {
    "TreeSum_SetB": [
        os.path.join(BASE_DIR, "mega_ablation_part1/summaries_alpha_1.0.json"),
        os.path.join(BASE_DIR, "mega_ablation_part2/summaries_alpha_1.0.json")
    ],
    "Flat_1024_SetA": [
        os.path.join(BASE_DIR, "Flat 1024/results_flat_1024/summaries_flat_1024.json")
    ],
    "Flat_Overlap_SetA": [
        os.path.join(BASE_DIR, "Flat 1024 Overlap/results_flat_overlap/summaries_flat_overlap.json")
    ],
    "TreeSum_SetA_Partial": [
        os.path.join(BASE_DIR, f"results_treesum_ablation_part1/summaries_batch_{i}.json") for i in range(1, 11)
    ]
}

def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def consolidate():
    # 1. Load data into a unified mapping
    master_map = {} # sample_id -> {method: summary, document: ..., reference: ...}
    
    variation_mapping = {
        "TreeSum_SetB": "treesum",
        "TreeSum_SetA_Partial": "treesum",
        "Flat_1024_SetA": "flat_1024",
        "Flat_Overlap_SetA": "flat_overlap"
    }
    
    for variation, files in PATHS.items():
        print(f"Processing {variation}...")
        method_key = variation_mapping[variation]
        
        for file_path in files:
            data = load_json(file_path)
            if data is None: continue
            
            for item in data:
                s_id = item.get('sample_id') or item.get('sample_idx')
                if s_id is None: continue
                
                if s_id not in master_map:
                    master_map[s_id] = {
                        "document": item.get("document", ""),
                        "reference": item.get("reference_summary", "")
                    }
                
                # Assign summary
                master_map[s_id][method_key] = item.get("generated_summary", "")
                
                # Fill missing doc/ref if found in later files
                if not master_map[s_id]["document"] and item.get("document"):
                    master_map[s_id]["document"] = item["document"]
                if not master_map[s_id]["reference"] and item.get("reference_summary"):
                    master_map[s_id]["reference"] = item["reference_summary"]

    # 2. Define Sample Order
    set_a_indices = load_json(INDEX_FILE_SET_A) or []
    other_indices = [s_id for s_id in master_map if s_id not in set_a_indices]
    
    ordered_ids = set_a_indices + sorted(other_indices)
    print(f"Total unique samples: {len(ordered_ids)}")

    # 3. Generate Final Report
    rows = []
    for s_id in ordered_ids:
        data = master_map[s_id]
        row = {
            "sample_id": s_id,
            "document": data["document"],
            "reference": data["reference"],
            "treesum": data.get("treesum", ""),
            "flat_1024": data.get("flat_1024", ""),
            "flat_overlap": data.get("flat_overlap", "")
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(BASE_DIR, "consolidated_1000_sample_ablation.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved master consolidation to {csv_path}")
    
    # Save individual JSONs for each variation matching the master csv (for Phase 17)
    for method in ["treesum", "flat_1024", "flat_overlap"]:
        method_data = []
        for s_id in ordered_ids:
            if master_map[s_id].get(method):
                method_data.append({
                    "sample_id": s_id,
                    "generated_summary": master_map[s_id][method],
                    "reference_summary": master_map[s_id]["reference"],
                    "document": master_map[s_id]["document"]
                })
        
        json_path = os.path.join(BASE_DIR, f"consolidated_{method}.json")
        with open(json_path, 'w') as f:
            json.dump(method_data, f, indent=2)
        print(f"Saved {len(method_data)} summaries to {json_path}")

if __name__ == "__main__":
    consolidate()
