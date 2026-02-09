
import os
import sys

# PART 1: Samples 0-500
PART_NAME = "part1"
START_IDX = 0
END_IDX = 500

# Optimization for P100: Use small batch sizes
P100_GEN_BATCH_SIZE = 4
P100_BERT_BATCH_SIZE = 8

def generate_partition_script(part_name, start_idx, end_idx, gen_batch, bert_batch):
    source_file = "/Users/mrarnav69/Documents/TreeSum/Production Kaggle/kaggle_complete_experiment-3.py"
    target_file = f"/Users/mrarnav69/Documents/TreeSum/Production Kaggle/run_treesum_{part_name}_500.py"
    
    with open(source_file, 'r') as f:
        content = f.read()
    
    # 1. Update Configuration for P100
    content = content.replace("GEN_BATCH_SIZE = 32", f"GEN_BATCH_SIZE = {gen_batch}")
    content = content.replace("BERT_BATCH_SIZE = 64", f"BERT_BATCH_SIZE = {bert_batch}")
    content = content.replace("DTYPE = torch.float32", "DTYPE = torch.float32") # Already set
    
    # 2. Update Output Directory for Kaggle
    kaggle_output = f"OUTPUT_DIR = '/kaggle/working/treesum_{part_name}_results'"
    content = content.replace("OUTPUT_DIR = os.path.join(BASE_DIR, 'results', 'treesum_production_results')", kaggle_output)
    
    # 3. Update Sample Selection for Partitioning
    partition_logic = f"""
    # Deterministic selection matching Flat baselines
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    
    # Partitioning: {part_name} ({start_idx} - {end_idx})
    indices = indices[{start_idx}:{end_idx}]
    """
    
    old_selection = """
    # Deterministic selection matching Flat baselines
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    """
    
    content = content.replace(old_selection, partition_logic)
    
    with open(target_file, 'w') as f:
        f.write(content)
    print(f"✅ Created {target_file}")

# Generate both parts
generate_partition_script("part1", 0, 500, P100_GEN_BATCH_SIZE, P100_BERT_BATCH_SIZE)
generate_partition_script("part2", 500, 1000, P100_GEN_BATCH_SIZE, P100_BERT_BATCH_SIZE)
