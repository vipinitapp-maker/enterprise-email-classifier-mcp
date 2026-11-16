"""Low-latency Email Classifier with Redis Caching, tiktoken Budgeting, and Agentic Routing."""
import hashlib
import json
import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import EmailTextCNN
from src.tokenizer import EmailTokenizer

AGENT_ROUTING_TABLE = {
    "DPDP_Compliance_Request": {"target_agent": "lexguard_compliance_agent", "requires_hitl_gate": True},
    "Security_Incident": {"target_agent": "aiops_soc_sentinel_agent", "requires_hitl_gate": True},
    "Customer_Escalation": {"target_agent": "priority_support_agent", "requires_hitl_gate": False},
    "Billing_Invoice": {"target_agent": "finance_erp_agent", "requires_hitl_gate": False},
    "General_Inquiry": {"target_agent": "self_service_rag_agent", "requires_hitl_gate": False},
}


class EmailTriageEngine:
    def __init__(self, ckpt_path: str = "artifacts/best_email_cnn.pt", tok_path: str = "artifacts/tokenizer.json"):
        if not os.path.exists(ckpt_path):
            from src.train import run_optuna_study
            run_optuna_study(trials=4, epochs=8)

        self.tokenizer = EmailTokenizer.load(tok_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.classes = ckpt["classes"]
        params = ckpt["params"]

        self.model = EmailTextCNN(
            vocab_size=ckpt["vocab_size"],
            embed_dim=params["embed_dim"],
            num_filters=params["num_filters"],
            num_classes=len(self.classes),
            dropout=params["dropout"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Optional Redis state cache with graceful in-memory fallback
        self._mem_cache = {}
        try:
            import redis
            self.redis_client = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)
            self.redis_client.ping()
        except Exception:
            self.redis_client = None

    def classify(self, subject: str, body: str) -> dict:
        full_text = f"{subject}. {body}".strip()
        cache_key = "email_triage:" + hashlib.sha256(full_text.encode("utf-8")).hexdigest()

        if self.redis_client:
            cached = self.redis_client.get(cache_key)
            if cached:
                res = json.loads(cached)
                res["cache_hit"] = True
                return res
        elif cache_key in self._mem_cache:
            res = dict(self._mem_cache[cache_key])
            res["cache_hit"] = True
            return res

        encoded = torch.tensor(self.tokenizer.encode(full_text)).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(self.model(encoded), dim=1).squeeze(0)
            conf, pred_idx = torch.max(probs, dim=0)

        category = self.classes[int(pred_idx)]
        routing = AGENT_ROUTING_TABLE[category]
        result = {
            "category": category,
            "confidence": round(float(conf), 4),
            "llm_token_count": self.tokenizer.count_llm_tokens(full_text),
            "target_agent": routing["target_agent"],
            "requires_hitl_gate": routing["requires_hitl_gate"],
            "cache_hit": False,
        }

        if self.redis_client:
            self.redis_client.setex(cache_key, 3600, json.dumps(result))
        else:
            self._mem_cache[cache_key] = result

        return result


if __name__ == "__main__":
    engine = EmailTriageEngine()
    samples = [
        ("DPDP Erasure Request", "Under DPDP Clause 11 please erase all my personal records immediately."),
        ("Production Export Outage", "P1 escalation: CSV export pipeline is timing out for enterprise audit."),
        ("SOC Alert", "Suspicious unauthorized API token activity detected on production cluster 4021."),
    ]
    for subj, msg in samples:
        print(subj, "->", json.dumps(engine.classify(subj, msg), indent=2))
