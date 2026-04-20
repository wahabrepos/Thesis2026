# Quick Start Guide - Self-MedRAG on Jetson Orin Nano

## ⚡ 5-Minute Setup

### Step 1: Clone & Setup
```bash
git clone <your-repo>
cd self-medrag-production

# Automated setup
chmod +x setup_jetson.sh
./setup_jetson.sh
```

### Step 2: Configure
```bash
# Edit environment variables
cp .env.template .env
nano .env

# Set these paths:
# PUBMED_CORPUS_PATH=./data/corpus/pubmed_abstracts.jsonl
# MEDQA_TRAIN_PATH=./data/datasets/medqa_train.json
```

### Step 3: Download Data
```bash
# Download datasets (10-15 minutes)
python scripts/download_datasets.py

# Download corpus (15-20 minutes, ~5GB)
python scripts/download_corpus.py --max-docs 50000
```

### Step 4: Test
```bash
# System check
python scripts/system_check.py

# Test query
python scripts/run_experiment.py --test-mode
```

### Step 5: Run
```bash
# Interactive mode
python scripts/query.py --interactive

# Run experiment
python scripts/run_experiment.py --dataset medqa --samples 100
```

## 📊 Expected Output

```
Query: What is the treatment for hypertension?
Answer: First-line treatment includes ACE inhibitors, diuretics, and lifestyle modifications.
Confidence: 0.87
Support Score: 0.82
Iterations: 2
Time: 8.3s
```

## 🔧 Troubleshooting

### Out of Memory?
Edit `config.yaml`:
```yaml
system:
  quantization: "8bit"  # More aggressive
  max_memory_gb: 3.5
```

### Slow?
```yaml
iteration:
  max_iterations: 1  # Disable refinement
generator:
  max_new_tokens: 256  # Shorter responses
```

### No CUDA?
```yaml
system:
  device: "cpu"  # Fallback to CPU (slow)
```

## 📚 Documentation

- Full setup: `SETUP_GUIDE.md`
- Configuration: `config.yaml` (all settings)
- API usage: See `src/pipeline.py`

## 🆘 Support

- Check system: `python scripts/system_check.py`
- View logs: `tail -f logs/medrag_*.log`
- GitHub issues: [your-repo/issues]
