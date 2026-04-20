# Deployment Checklist - Jetson Orin Nano

## Pre-Deployment

- [ ] Jetson Orin Nano Super with JetPack 6.0+ installed
- [ ] Ubuntu 22.04 LTS running
- [ ] 128GB+ storage (64GB minimum)
- [ ] Active cooling fan installed
- [ ] External SSD connected (optional but recommended)
- [ ] Network connectivity for downloads

## Installation

- [ ] Run `setup_jetson.sh` successfully
- [ ] All dependencies installed (`pip list | grep torch`)
- [ ] CUDA detected (`python -c "import torch; print(torch.cuda.is_available())"`)
- [ ] System check passed (`python scripts/system_check.py`)

## Configuration

- [ ] `.env` file created from template
- [ ] Paths configured in `.env`:
  - [ ] `HF_HOME` (model cache)
  - [ ] `PUBMED_CORPUS_PATH`
  - [ ] `MEDQA_TRAIN_PATH`
  - [ ] `PUBMEDQA_TRAIN_PATH`
- [ ] `config.yaml` reviewed and adjusted for your hardware
- [ ] API keys added (if using fallback mode)

## Data Download

- [ ] MedQA dataset downloaded (`ls data/datasets/medqa_*.json`)
- [ ] PubMedQA dataset downloaded (`ls data/datasets/pubmedqa_*.json`)
- [ ] Corpus downloaded (`wc -l data/corpus/pubmed_abstracts.jsonl`)
- [ ] Models cached (`ls models/` or check `~/.cache/huggingface/`)

## Testing

- [ ] Test mode passed (`python scripts/run_experiment.py --test-mode`)
- [ ] Single query works (`python scripts/query.py -q "What is diabetes?"`)
- [ ] Interactive mode works (`python scripts/query.py -i`)
- [ ] Benchmark run completed (`python scripts/run_experiment.py --dataset medqa --samples 10`)

## Performance Validation

- [ ] VRAM usage < 7GB during inference
- [ ] Latency P50 < 15 seconds
- [ ] No out-of-memory errors
- [ ] GPU utilization > 80% during generation
- [ ] Accuracy on test set > 80%

## Production Readiness

- [ ] Logging configured and working (`tail logs/medrag_*.log`)
- [ ] Checkpointing tested (interrupt and resume)
- [ ] Error recovery tested (intentional failure)
- [ ] Monitoring metrics exported (`results/metrics_*.json`)
- [ ] Documentation accessible to team

## Optional Enhancements

- [ ] Weights & Biases integration (set `WANDB_API_KEY`)
- [ ] Slack alerts configured (set `SLACK_WEBHOOK_URL`)
- [ ] Fine-tuned model on medical data
- [ ] API mode deployed (Flask/FastAPI wrapper)
- [ ] Load balancer for multiple Jetsons

## Maintenance

- [ ] Backup checkpoints regularly
- [ ] Monitor disk space (models are large)
- [ ] Update corpus monthly
- [ ] Review error logs weekly
- [ ] Update models quarterly

## Security

- [ ] `.env` file not committed to git
- [ ] API keys stored securely
- [ ] Network access restricted
- [ ] Medical data compliance (HIPAA if applicable)

## Sign-Off

- [ ] Deployment verified by: ________________
- [ ] Date: ________________
- [ ] Performance meets SLA: ☐ Yes ☐ No
- [ ] Ready for production: ☐ Yes ☐ No

---

**Notes:**
- Keep this checklist updated as system evolves
- Document any deviations from standard setup
- Share lessons learned with team
