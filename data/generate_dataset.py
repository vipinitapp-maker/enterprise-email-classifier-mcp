"""Generates a realistic enterprise email & DPDP compliance dataset for training."""
import json
import os
import random

CLASSES = [
    "DPDP_Compliance_Request",
    "Security_Incident",
    "Customer_Escalation",
    "Billing_Invoice",
    "General_Inquiry",
]

TEMPLATES = {
    "DPDP_Compliance_Request": [
        "Under the Digital Personal Data Protection Act DPDP Clause 11 please erase all my personal records from tenant {id}.",
        "Data principal request: I withdraw consent for marketing processing and demand right to correction of profile {id}.",
        "Statutory notice under DPDP Section 5: provide complete audit log and summary of personal data stored for user {id}.",
        "Please execute mandatory cryptographic purge of my PII data and confirm deletion timeline per DPDP compliance rules.",
        "Requesting nomination update and data access report under DPDP Act data principal rights for account {id}.",
    ],
    "Security_Incident": [
        "URGENT: Suspicious unauthorized API token activity and anomalous outbound traffic detected on production cluster {id}.",
        "Security alert: Possible PII data leakage and ACL permission bypass observed in export service logs.",
        "Phishing campaign targeting finance mailboxes with malicious attachment hash reported by SOC team.",
        "Critical vulnerability CVE alert: Immediate sandbox isolation required for agent worker node {id}.",
        "Failed SSO login spike and credential stuffing attempt blocked at gateway firewall.",
    ],
    "Customer_Escalation": [
        "Production outage P1: Enterprise export jobs are failing with timeout errors blocking our quarterly audit.",
        "Escalating ticket {id}: Multi-agent workflow latency exceeded SLA threshold for the past 4 hours.",
        "Executive escalation: Client onboarding pipeline stalled at approval gate; need immediate engineering intervention.",
        "Repeated regression in CSV export formatting is impacting downstream ERP ingestion for account {id}.",
        "High priority support escalation: Webhook delivery failures causing dropped customer notifications.",
    ],
    "Billing_Invoice": [
        "Please find attached GST tax invoice {id} for cloud compute and monthly SaaS subscription renewal.",
        "Payment confirmation and remittance advice for PO {id} processed via NEFT.",
        "Requesting credit note and revised pro-forma invoice for seat downgrade on enterprise plan.",
        "Accounts payable reminder: Invoice {id} is due within 7 business days; confirm bank details.",
        "Query regarding token overage charges and tier pricing breakdown on October billing statement.",
    ],
    "General_Inquiry": [
        "Hi team, can we schedule a product walkthrough demo for our architecture review board next Tuesday?",
        "Looking for documentation on configuring custom webhook endpoints and role permissions.",
        "Thank you for sharing the release notes and updated integration guide for v2.4.",
        "Could you share availability for a quick sync regarding upcoming Q4 roadmap milestones?",
        "Following up on our earlier discussion about sandbox environment provisioning.",
    ],
}


def build_dataset(samples_per_class: int = 240, seed: int = 42) -> list[dict]:
    random.seed(seed)
    records = []
    for label_idx, label_name in enumerate(CLASSES):
        base_templates = TEMPLATES[label_name]
        for i in range(samples_per_class):
            tpl = random.choice(base_templates)
            text = tpl.format(id=random.randint(1000, 9999))
            # Add light realistic noise/tokens
            suffix = random.choice([
                "",
                " Regards, Enterprise Ops.",
                " Please prioritize and update ticket status.",
                " Ref: #YO-" + str(random.randint(100, 999)),
            ])
            records.append({
                "text": text + suffix,
                "label": label_idx,
                "category": label_name,
            })
    random.shuffle(records)
    return records


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dataset = build_dataset(samples_per_class=240)
    out_path = "data/emails_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)
    print(f"Generated {len(dataset)} labeled email records -> {out_path}")
