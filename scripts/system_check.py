#!/usr/bin/env python3
"""
System Check - Verify Installation
"""

import sys
from pathlib import Path


def check_python():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ✗ Python 3.8+ required")
        return False
    print("  ✓ OK")
    return True


def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"CUDA available: {cuda_available}")
            print(f"  GPU: {device_name}")
            print(f"  Memory: {memory:.1f}GB")
            print("  ✓ OK")
        else:
            print("CUDA available: False")
            print("  ⚠ Running on CPU (slow)")
        
        return True
    except Exception as e:
        print(f"CUDA check failed: {e}")
        print("  ✗ FAIL")
        return False


def check_dependencies():
    """Check required packages."""
    required = [
        "torch",
        "transformers",
        "sentence_transformers",
        "faiss",
        "rank_bm25",
        "pandas",
        "numpy",
        "sklearn",
        "yaml",
        "dotenv",
    ]
    
    print("Checking dependencies:")
    all_ok = True
    
    for package in required:
        try:
            if package == "faiss":
                __import__("faiss")
            elif package == "yaml":
                __import__("yaml")
            elif package == "dotenv":
                __import__("dotenv")
            elif package == "sklearn":
                __import__("sklearn")
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            all_ok = False
    
    return all_ok


def check_files():
    """Check required files exist."""
    project_root = Path(__file__).parent.parent
    
    required_files = [
        "config.yaml",
        ".env.template",
        "src/__init__.py",
        "src/pipeline.py",
        "src/model.py",
        "src/retrieval.py",
    ]
    
    print("Checking files:")
    all_ok = True
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_ok = False
    
    return all_ok


def check_directories():
    """Check required directories."""
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "data/datasets",
        "data/corpus",
        "models",
        "results",
        "logs",
    ]
    
    print("Checking directories:")
    all_ok = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ⚠ {dir_path} (will be created)")
            full_path.mkdir(parents=True, exist_ok=True)
    
    return all_ok


def check_config():
    """Check configuration."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    
    try:
        from utils import load_config
        config = load_config("config.yaml")
        print("Configuration loaded:")
        print(f"  Generator: {config.get('generator', {}).get('model_name', 'unknown')}")
        print(f"  Retrieval: {config.get('retrieval', {}).get('dense', {}).get('model_name', 'unknown')}")
        print("  ✓ OK")
        return True
    except Exception as e:
        print(f"Configuration check failed: {e}")
        print("  ✗ FAIL")
        return False


def main():
    print("="*60)
    print("Self-MedRAG System Check")
    print("="*60)
    print()
    
    checks = []
    
    checks.append(("Python Version", check_python()))
    print()
    
    checks.append(("CUDA/GPU", check_cuda()))
    print()
    
    checks.append(("Dependencies", check_dependencies()))
    print()
    
    checks.append(("Files", check_files()))
    print()
    
    checks.append(("Directories", check_directories()))
    print()
    
    checks.append(("Configuration", check_config()))
    print()
    
    # Summary
    print("="*60)
    print("Summary:")
    print("="*60)
    
    for name, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20} {status}")
    
    all_passed = all(passed for _, passed in checks)
    
    print()
    if all_passed:
        print("✓ All checks passed! System ready.")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
