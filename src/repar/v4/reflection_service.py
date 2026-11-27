"""Optional inference service for the existing post-observation reflection."""
from repar.model_support import BoundedLLMClient


def reflection_client(client, route):
    """Create a separate client with the same model, seed and sampling."""
    config = {
        **client.config,
        "base_url": route["base_url"],
        "deployment_id": route["deployment_id"] + "/" + client.config.get("deployment_id", client.base),
        "enable_thinking": route["enable_thinking"],
    }
    return BoundedLLMClient(config, client.cache_dir, deadline=getattr(client, "deadline", None),
                            call_seconds=getattr(client, "call_seconds", 1200))
