import os
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import shutil

repos = {
    'alignscore': {
        'url': 'https://github.com/yuh-zha/AlignScore/archive/refs/heads/main.zip',
        'extract_name': 'AlignScore-main',
        'target_name': 'AlignScore'
    },
    'bartscore': {
        'url': 'https://github.com/neulab/BARTScore/archive/refs/heads/main.zip',
        'extract_name': 'BARTScore-main', 
        'target_name': 'BARTScore'
    },
    'unieval': {
        'url': 'https://github.com/maszhongming/UniEval/archive/refs/heads/main.zip',
        'extract_name': 'UniEval-main',
        'target_name': 'UniEval'
    }
}

models = {
    'alignscore-large': {
        'url': 'https://huggingface.co/nyu-mll/AlignScore-large/resolve/main/AlignScore-large.ckpt',
        'filename': 'AlignScore-large.ckpt'
    }
}

def download_file(url, filename, description="file"):
    print(f"downloading {description}...")
    try:
        urlretrieve(url, filename)
        print(f"successfully downloaded {description}")
        return True
    except Exception as e:
        print(f"failed to download {description}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    print(f"extracting {zip_path}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("extraction complete")
        return True
    except Exception as e:
        print(f"failed to extract {zip_path}: {e}")
        return False

def setup_repository(repo_name, repo_info):
    print(f"setting up {repo_name}...")
    
    zip_filename = f"{repo_name}.zip"
    if not download_file(repo_info['url'], zip_filename, repo_name):
        return False
    
    if not extract_zip(zip_filename, "."):
        return False
    
    extract_path = Path(repo_info['extract_name'])
    target_path = Path(repo_info['target_name'])
    
    if extract_path.exists():
        if target_path.exists():
            shutil.rmtree(target_path)
        shutil.move(str(extract_path), str(target_path))
        print(f"{repo_name} setup complete")
        os.remove(zip_filename)
        return True
    else:
        print(f"error: extracted directory {extract_path} not found")
        return False

def setup_models():
    print("setting up models...")
    for model_name, model_info in models.items():
        if not download_file(model_info['url'], model_info['filename'], model_name):
            print(f"failed to download {model_name}")
        else:
            print(f"{model_name} downloaded successfully")

def check_dependencies():
    print("checking dependencies...")
    required_packages = ['zipfile', 'shutil', 'pathlib']
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"missing packages: {missing}")
        return False
    
    print("all dependencies available")
    return True

def main():
    print("evaluation repository setup")
    print("-" * 30)
    
    if not check_dependencies():
        return False
    
    success_count = 0
    for repo_name, repo_info in repos.items():
        if setup_repository(repo_name, repo_info):
            success_count += 1
    
    print(f"\nrepositories setup: {success_count}/{len(repos)}")
    setup_models()
    print("\nsetup complete")
    
    return True

if __name__ == "__main__":
    main()
