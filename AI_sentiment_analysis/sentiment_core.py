"""
sentiment_core.py
=================

Shared sentiment-analysis primitives: the output schema, the prompt loader,
the Omeka content extractor and the per-model / all-model analysis calls.

Both the production run (``01_sentiment_analysis.py``) and the pilot
(``02_pilot_new_panel.py``) import from here. That matters more than the usual
DRY argument: a pilot whose schema or prompt has drifted from production is not
comparable to it, and the whole point of the pilot is comparability.

Nothing in this module writes to Omeka.
"""
import time
import logging
import concurrent.futures
from pathlib import Path
from typing import Optional, Dict, Any, Literal

from pydantic import BaseModel, Field

from common.llm_provider import BaseLLMClient

# ---------------------------------------------------------------------------
# Controlled vocabulary
# ---------------------------------------------------------------------------

CENTRALITE_LABELS = ("Très central", "Central", "Secondaire", "Marginal", "Non abordé")
POLARITE_LABELS = (
    "Très positif", "Positif", "Neutre", "Négatif", "Très négatif", "Non applicable",
)

#: Ordinal rank of each label, weakest→strongest. Used by the pilot's agreement
#: maths; ``Non applicable`` has no place on the scale and is deliberately absent.
POLARITE_ORDER = {
    "Très négatif": 1, "Négatif": 2, "Neutre": 3, "Positif": 4, "Très positif": 5,
}
CENTRALITE_ORDER = {
    "Non abordé": 1, "Marginal": 2, "Secondaire": 3, "Central": 4, "Très central": 5,
}

# Omeka items backing the controlled vocabulary. Centralité, polarité and the
# subjectivité score are all stored as resource:item links, not literals.
# (Instance-specific, so arguably common/iwac_config.py — kept here because
# only the sentiment scripts use them.)
CENTRALITE_ITEM_IDS = {
    "Très central": 78048,
    "Central": 78049,
    "Secondaire": 78050,
    "Marginal": 78051,
    "Non abordé": 78052,
}

POLARITE_ITEM_IDS = {
    "Très positif": 78031,
    "Positif": 78038,
    "Neutre": 78039,
    "Négatif": 78040,
    "Très négatif": 78041,
    "Non applicable": 78042,
}

SUBJECTIVITE_ITEM_IDS = {
    1: 78043,  # Très objectif
    2: 78044,  # Plutôt objectif
    3: 78045,  # Mixte
    4: 78046,  # Plutôt subjectif
    5: 78047,  # Très subjectif
}

#: Reverse map for reading a stored subjectivité link back to its 1-5 score.
ITEM_ID_TO_SUBJECTIVITE = {v: k for k, v in SUBJECTIVITE_ITEM_IDS.items()}


class SentimentAnalysisOutput(BaseModel):
    """Schema for sentiment analysis output - used for structured outputs with AI APIs."""
    centralite_islam_musulmans: Literal[
        "Très central", "Central", "Secondaire", "Marginal", "Non abordé"
    ] = Field(description="Importance accordée aux thèmes liés à l'islam et aux musulmans dans l'article")
    centralite_justification: str = Field(description="Courte justification en 1 phrase sur la centralité de l'islam/des musulmans")
    subjectivite_score: Optional[int] = Field(
        default=None,
        description="Score de subjectivité de 1 à 5 (entier), ou null si le sujet n'est pas abordé",
    )
    subjectivite_justification: str = Field(description="Justification en 1-2 phrases pour le score de subjectivité")
    polarite: Literal[
        "Très positif", "Positif", "Neutre", "Négatif", "Très négatif", "Non applicable"
    ] = Field(description="Sentiment général exprimé dans l'article envers l'islam et/ou les musulmans")
    polarite_justification: str = Field(description="Justification en 1-2 phrases pour la polarité")


# ---------------------------------------------------------------------------
# Prompt + content
# ---------------------------------------------------------------------------

PROMPT_FILENAME = "sentiment_prompt.md"


def load_system_prompt() -> str:
    """Load the sentiment analysis prompt from the markdown file.

    Raises rather than returning "" — a silent empty prompt would produce
    plausible-looking but unanchored annotations.
    """
    prompt_path = Path(__file__).resolve().parent / PROMPT_FILENAME
    return prompt_path.read_text(encoding="utf-8")


def create_user_prompt(article_text: str) -> str:
    """Create the user prompt with the article text to analyze."""
    return f"Texte à analyser:\n---\n{article_text}\n---"


def get_item_content(item: Dict[str, Any]) -> str:
    """Extract bibo:content text from an Omeka item."""
    content_values = item.get('bibo:content', [])
    if not content_values:
        return ""

    # Prefer French language if available
    for val in content_values:
        if isinstance(val, dict) and val.get('@language') == 'fr':
            return val.get('@value', '')

    # Fall back to first value
    if content_values and isinstance(content_values[0], dict):
        return content_values[0].get('@value', '')

    return ""


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_with_model(
    llm_client: BaseLLMClient,
    text: str,
    system_prompt: str,
    model_label: str,
    logger: logging.Logger,
    max_retries: int = 3
) -> Dict[str, Any]:
    """Analyze text using any LLM provider via the shared provider."""
    default_error = {
        "centralite_islam_musulmans": "ERREUR_ANALYSE",
        "centralite_justification": f"Erreur lors de l'analyse {model_label}.",
        "subjectivite_score": None,
        "subjectivite_justification": f"Erreur lors de l'analyse {model_label}.",
        "polarite": "ERREUR_ANALYSE",
        "polarite_justification": f"Erreur lors de l'analyse {model_label}.",
        "analysis_error": "Unknown error"
    }

    if not text or not text.strip():
        return {
            **default_error,
            "centralite_islam_musulmans": "Non abordé",
            "centralite_justification": "Texte non fourni ou vide.",
            "subjectivite_justification": "Non applicable - texte vide.",
            "polarite": "Non applicable",
            "polarite_justification": "Non applicable - texte vide.",
            "analysis_error": "Empty text"
        }

    user_prompt = create_user_prompt(text)
    last_error = "Max retries exceeded"

    for attempt in range(max_retries):
        try:
            result = llm_client.generate_structured(
                system_prompt, user_prompt, SentimentAnalysisOutput
            )
            return {**result.model_dump(), "analysis_error": None}
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.debug(f"{model_label} attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    # Keep the real error: "Max retries exceeded" alone makes a bad API key and
    # a malformed response look identical when triaging a failed run.
    return {**default_error, "analysis_error": last_error}


def _timeout_result(reason: str) -> Dict[str, Any]:
    return {
        "analysis_error": reason,
        "centralite_islam_musulmans": "ERREUR_ANALYSE",
        "centralite_justification": f"Erreur: {reason}",
        "subjectivite_score": None,
        "subjectivite_justification": f"Erreur: {reason}",
        "polarite": "ERREUR_ANALYSE",
        "polarite_justification": f"Erreur: {reason}",
    }


def analyze_with_all_models(
    text: str,
    llm_clients: Dict[str, BaseLLMClient],
    system_prompt: str,
    logger: logging.Logger,
    labels: Optional[Dict[str, str]] = None,
    timeout: int = 120,
) -> Dict[str, Dict[str, Any]]:
    """Run sentiment analysis with all supplied models concurrently.

    Returns:
        Dictionary with model names as keys and results as values
    """
    labels = labels or {}
    results: Dict[str, Dict[str, Any]] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(llm_clients)) as executor:
        futures = {}

        for model_name, llm_client in llm_clients.items():
            label = labels.get(model_name, model_name.capitalize())
            futures[model_name] = executor.submit(
                analyze_with_model, llm_client, text, system_prompt, label, logger
            )

        for model_name, future in futures.items():
            try:
                results[model_name] = future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                results[model_name] = _timeout_result("Timeout")
            except Exception as e:
                results[model_name] = _timeout_result(f"{type(e).__name__}: {e}")

    return results


def is_valid_result(result: Dict[str, Any]) -> bool:
    """True when a result carries a usable annotation rather than an error."""
    if result.get("analysis_error"):
        return False
    return result.get("centralite_islam_musulmans") != "ERREUR_ANALYSE"
