```markdown
# Enterprise Email Classification & Deep Learning Triage Pipeline (PyTorch + Optuna + MLflow + MCP)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-HPO-blueviolet.svg)](https://optuna.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194e2.svg)](https://mlflow.org/)
[![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-10b981.svg)](https://modelcontextprotocol.io/)

An end-to-end, production-grade AI/ML and Agentic Orchestration pipeline designed to solve a core enterprise bottleneck: **routing every incoming email straight to an LLM introduces high latency, unnecessary token burn, and non-deterministic compliance risks.**

Instead, this system uses a locally trained **PyTorch 1D Convolutional Neural Network (TextCNN)**—tuned automatically via **Optuna** and tracked in **MLflow**—to classify emails in **<5ms**, cache repeated payloads in **Redis**, measure downstream LLM token budgets with **`tiktoken`**, and expose deterministic routing + **Human-in-the-Loop (HITL)** gate policies to multi-agent hosts via a **Model Context Protocol (MCP)** server.

---

## System Architecture

```text
                       ┌──────────────────────────────────────────┐
                       │   Incoming Enterprise Email (Subject/Body)│
                       └────────────────────┬─────────────────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │  MCP Tool Server (src/mcp_server.py)     │
                       │  • Tool: classify_and_route_email        │
                       │  • Resource: triage://routing-policy     │
                       └────────────────────┬─────────────────────┘
                                            │
                       ┌────────────────────▼─────────────────────┐
                       │  Redis SHA-256 Deduplication Cache       │
                       │  (Graceful in-memory fallback)           │
                       └──────────┬────────────────────┬──────────┘
                         Cache Hit│                    │Cache Miss
                                  │                    ▼
                                  │   ┌───────────────────────────────────┐
                                  │   │ Tokenizer & tiktoken Budgeter     │
                                  │   │ (cl100k_base token cost estimate) │
                                  │   └────────────────┬──────────────────┘
                                  │                    │
                                  │                    ▼
                                  │   ┌───────────────────────────────────┐
                                  │   │ PyTorch 1D TextCNN Classifier     │
                                  │   │ Parallel N-Gram Filters (2, 3, 4) │
                                  │   │ + Global Max-Over-Time Pooling    │
                                  │   └────────────────┬──────────────────┘
                                  │                    │
                                  └─────────┬──────────┘
                                            │
                                            ▼
                       ┌──────────────────────────────────────────┐
                       │   Deterministic Agent & HITL Router      │
                       ├──────────────────────┬───────────────────┤
                       │ High-Risk / Regulated│ Standard Operational
                       │ (HITL Gate = True)   │ (HITL Gate = False)
                       ▼                      ▼
           ┌───────────────────────┐  ┌───────────────────────────┐
           │ • LexGuard DPDP Agent │  │ • Priority Support Agent  │
           │ • AIOps SOC Sentinel  │  │ • Finance / ERP Agent     │
           │   (Admin Approval)    │  │ • Self-Service RAG Agent  │
           └───────────────────────┘  └───────────────────────────┘

```

---

## Agentic Triage & HITL Routing Policy

Each classified email is deterministically mapped to a specialized downstream agent along with a **Human-in-the-Loop (HITL)** governance flag and exact `tiktoken` payload token count:

| Predicted Intent Class | Target Downstream Agent | HITL Approval Gate | Example Trigger Signals |
| --- | --- | --- | --- |
| **`DPDP_Compliance_Request`** | `lexguard_compliance_agent` | **Required (`True`)** | DPDP Clause 11 erasure, Clause 5 notices, consent withdrawal |
| **`Security_Incident`** | `aiops_soc_sentinel_agent` | **Required (`True`)** | Unauthorized API tokens, PII leakage alerts, credential stuffing |
| **`Customer_Escalation`** | `priority_support_agent` | Auto (`False`) | P1 production outages, SLA latency breaches, export failures |
| **`Billing_Invoice`** | `finance_erp_agent` | Auto (`False`) | GST tax invoices, PO remittance, token overage statements |
| **`General_Inquiry`** | `self_service_rag_agent` | Auto (`False`) | Product walkthroughs, webhook setup docs, roadmap syncs |

---

## Deep Learning & MLOps Design

* **Multi-Kernel 1D TextCNN (`src/model.py`):**
* **Embedding Layer:** Maps normalized vocabulary tokens (`max_vocab_size=5000`, `max_seq_len=48`) into dense embeddings.
* **Parallel 1D Convolutions:** Extracts bigram ($k=2$), trigram ($k=3$), and 4-gram ($k=4$) phrase features simultaneously, followed by `ReLU` and global max-over-time pooling (`F.max_pool1d`) for position-invariant signal extraction.
* **Regularization:** Concatenates pooled n-gram feature maps and applies `Dropout` prior to the linear classification head.


* **Automated Hyperparameter Tuning & MLflow Tracking (`src/train.py`):**
* Runs an **Optuna** maximization study over `embed_dim` (`32, 64, 128`), `num_filters` (`32, 64, 128`), `dropout` (`0.15–0.50`), log-uniform learning rate `lr` (`5e-4` to `5e-3`), and `batch_size` (`16, 32`).
* Logs every trial as a nested child run under a parent MLflow experiment (`Email-Classification-TextCNN`), promoting the winning checkpoint (`artifacts/best_email_cnn.pt`) and serialized PyTorch model artifact (`mlflow.pytorch.log_model`).


* **Production Resilience (`src/inference.py`):**
* Computes a `SHA-256` hash of `subject + body` to serve duplicate webhook retries from **Redis** (`TTL = 3600s`), with automatic fallback to in-process memory caching if Redis is unavailable.



---

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

---

## Quickstart

### 1. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
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

Inspect Optuna trial metrics, parameters, and versioned PyTorch artifacts in the MLflow UI:

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

#### Sample MCP Tool Response (`classify_and_route_email`)

```json
{
  "category": "DPDP_Compliance_Request",
  "confidence": 0.9984,
  "llm_token_count": 18,
  "target_agent": "lexguard_compliance_agent",
  "requires_hitl_gate": true,
  "cache_hit": false
}

```
