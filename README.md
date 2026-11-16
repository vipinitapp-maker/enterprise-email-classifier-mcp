# Enterprise Email Classification & Deep Learning Triage Pipeline (PyTorch + Optuna + MLflow + MCP)

An end-to-end, production-grade AI/ML repository demonstrating:
- **1D Convolutional Neural Network (TextCNN)** built in **PyTorch** and **NumPy** for multi-class enterprise email intent and regulatory compliance triage (`DPDP_Compliance_Request`, `Security_Incident`, `Customer_Escalation`, `Billing_Invoice`, `General_Inquiry`).
- **Automated Hyperparameter Optimization** using **Optuna** (`TPE` sampler + `MedianPruner`) with **MLflow** parent/child run tracking and model registry integration.
- **Token Budgeting & Context Optimization** using **`tiktoken`** prior to downstream LLM/agent escalation.
- **Model Context Protocol (MCP) Server** exposing the trained classifier and **Redis**-backed triage state/cache as standardized tools for multi-agent orchestration.

## Repository Structure

```text
enterprise-email-classifier-mcp/
├── data/
│   └── generate_dataset.py      # Synthetic enterprise email & DPDP compliance dataset generator
├── src/
│   ├── tokenizer.py             # Vocabulary builder & tiktoken token budget analyzer
│   ├── model.py                 # Multi-kernel 1D TextCNN architecture in PyTorch
│   ├── train.py                 # Optuna hyperparameter tuning + MLflow parent/child tracking
│   ├── inference.py             # Low-latency classifier with Redis caching & HITL routing
│   └── mcp_server.py            # Model Context Protocol (MCP) server exposing triage tools
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Enterprise Dataset
```bash
python data/generate_dataset.py
```

### 3. Run Hyperparameter Tuning & Training (Optuna + MLflow)
```bash
python src/train.py --trials 10 --epochs 12
```
View tracked Optuna trials, metrics, and logged PyTorch model artifacts in the MLflow UI:
```bash
mlflow ui --port 5000
```

### 4. Run Local Inference & Agentic Triage Demo
```bash
python src/inference.py
```

### 5. Launch the MCP Tool Server
```bash
mcp run src/mcp_server.py
```
