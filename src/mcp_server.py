"""Model Context Protocol (MCP) Server exposing the Email CNN Triage Engine to Multi-Agent Hosts."""
from mcp.server.fastmcp import FastMCP
from src.inference import AGENT_ROUTING_TABLE, EmailTriageEngine

mcp = FastMCP("EnterpriseEmailTriageMCP")
_engine = None


def get_engine() -> EmailTriageEngine:
    global _engine
    if _engine is None:
        _engine = EmailTriageEngine()
    return _engine


@mcp.tool()
def classify_and_route_email(subject: str, body: str) -> dict:
    """
    Classify an incoming enterprise email using the trained PyTorch TextCNN model,
    estimate LLM token cost via tiktoken, and return the target agent & HITL gate policy.
    """
    return get_engine().classify(subject=subject, body=body)


@mcp.resource("triage://routing-policy")
def get_routing_policy() -> str:
    """Expose the deterministic agent routing table and HITL approval rules."""
    return str(AGENT_ROUTING_TABLE)


if __name__ == "__main__":
    mcp.run()
