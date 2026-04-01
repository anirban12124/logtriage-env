# LogTriage — SRE Log Investigation Environment

An OpenEnv-compliant AI agent evaluation environment that simulates
real-world Site Reliability Engineer (SRE) log investigation during incidents.

## Overview

Every tech company has SREs who investigate system logs during outages
and security incidents. This environment simulates that process:
the agent reads logs, identifies anomalies, traces causal chains,
classifies severity, and produces an incident report.

The environment tests 7 cognitive dimensions:
- **Information Retrieval** — searching and filtering effectively
- **Pattern Recognition** — spotting anomalies in noisy data
- **Causal Reasoning** — tracing root cause → symptoms
- **Cross-referencing** — linking events across services
- **Prioritization** — focusing on what matters
- **Synthesis** — summarizing findings coherently
- **Classification** — assessing severity accurately

## Quick Start

```bash
# Build and run locally
docker build -t logtriage .
docker run -p 8000:8000 logtriage

# Or run directly
uvicorn app:app --host 0.0.0.0 --port 8000

# In another terminal, run the baseline agent
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model-name"
export HF_TOKEN="your-token"
export SPACE_URL="http://localhost:8000"
python inference.py