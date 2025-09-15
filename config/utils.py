from enum import Enum
from typing import Any, Dict, List


def get_base_contexts() -> List[str]:
    """
    Provides standard business analysis contexts for prompt generation.

    Returns a curated list of general business analysis categorie
    applicable across industries for context-aware prompt optimization.

    Returns:
        List of general business context categorie
    """
    return [
        "Business Analysis",
        "Market Research",
        "Strategic Planning",
        "Industry Insights",
    ]


def get_domain_specific_contexts() -> Dict[str, List[str]]:
    """
    Provides industry-specific context categories.

    Returns specialized context categories organized by industry
    domain to support domain-targeted prompt optimization.

    Returns:
        Dictionary mapping industry domains to relevant context
    """
    return {
        "retail": [
            "Consumer Behavior",
            "Retail Operations",
            "Supply Chain",
            "E-commerce",
        ],
        "technology": [
            "Product Innovation",
            "R&D Strategy",
            "Tech Stack",
            "Digital Transformation",
        ],
        "sustainability": [
            "ESG Metrics",
            "Green Initiative",
            "Carbon Footprint",
            "Ethical Sourcing",
        ],
    }


def get_context_weights() -> Dict[str, float]:
    """
    Provides weighting factors for different context types.

    Returns relative importance weights for different context
    categories to guide the prompt optimization process.

    Returns:
        Mapping of context types to importance weight
    """
    return {"base": 1.0, "domain": 0.8, "hybrid": 0.6}


def get_context_parameters() -> Dict[str, Any]:
    """
    Provides parameters for context application during optimization.

    Returns configuration parameters that control how context
    are selected and combined during prompt generation.

    Returns:
        Configuration parameters for context processing
    """
    return {
        "max_contexts_per_example": 3,
        "min_context_similarity": 0.7,
        "context_diversity_weight": 0.8,
        "domain_specificity_threshold": 0.6,
    }


def get_default_model_names() -> List[str]:
    """
    Provides the supported language model identifiers.

    Returns the list of available model identifiers that
    can be used for prompt optimization tasks.

    Returns:
        List of supported model identifier
    """
    # TODO get rid of hardcoded model name
    return [
        # "meta.llama3-2-3b-instruct-v1",
        "anthropic.claude-v3-5-sonnet",
        "anthropic.claude-v3-haiku",
        "anthropic.claude-v3-5-haiku",
    ]


def get_empty_dict() -> Dict[str, Any]:
    """
    Provides an empty dictionary for optional configurations.

    Returns an empty dictionary for initializing optional
    configuration objects with default empty values.

    Returns:
        Empty dictionary for default configuration
    """
    return {}


class RequestErrorHandler(str, Enum):
    """
    Enumeration of strategies for handling request errors.

    Defines standardized approaches for handling errors during
    API requests, particularly for retrieval operations.
    """

    RETRY_RAISE = "retry_and_rise"
    RETRY_FILL = "retry_and_fill"
