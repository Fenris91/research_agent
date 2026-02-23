"""Field enrichment pipeline: OpenAlex concepts → LLM fallback.

When papers arrive with poor or missing fields (common for S2-sourced papers),
this module enriches them by querying OpenAlex and falling back to LLM extraction.
"""

import json
import logging
import os
from typing import List, Optional

import httpx

from research_agent.utils.openalex import extract_openalex_fields
from research_agent.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

OPENALEX_API = "https://api.openalex.org"


def _openalex_email() -> str:
    """Return the configured email for OpenAlex polite pool access."""
    return os.environ.get("OPENALEX_EMAIL", "research-agent@example.com")

# Coarse S2 top-level categories that don't add much signal for the
# field→domain scaffold in the Knowledge Explorer.
_COARSE_S2_FIELDS = frozenset({
    "Art",
    "Biology",
    "Business",
    "Chemistry",
    "Computer Science",
    "Economics",
    "Education",
    "Engineering",
    "Environmental Science",
    "Geography",
    "Geology",
    "History",
    "Law",
    "Linguistics",
    "Materials Science",
    "Mathematics",
    "Medicine",
    "Philosophy",
    "Physics",
    "Political Science",
    "Psychology",
    "Sociology",
})

# ---------------------------------------------------------------------------
# Quality heuristic
# ---------------------------------------------------------------------------


def fields_need_enrichment(fields: Optional[List[str]]) -> bool:
    """Return True if *fields* are missing, empty, or all coarse S2 top-level labels."""
    if not fields:
        return True
    non_coarse = [f for f in fields if f not in _COARSE_S2_FIELDS]
    return len(non_coarse) == 0


# ---------------------------------------------------------------------------
# OpenAlex lookup
# ---------------------------------------------------------------------------


def _titles_match(a: str, b: str, threshold: float = 0.5) -> bool:
    """Jaccard similarity on lowercased word sets — True if >= *threshold*."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def _lookup_openalex_by_doi(doi: str) -> Optional[List[str]]:
    """Try OpenAlex /works/doi:{doi} and extract concepts."""
    try:
        resp = retry_with_backoff(
            lambda: httpx.get(
                f"{OPENALEX_API}/works/doi:{doi}",
                params={"mailto": _openalex_email()},
                headers={"User-Agent": "research-agent/1.0"},
                timeout=10,
            ),
            max_retries=2,
            base_delay=1.0,
            retry_on=(429, 503, 504),
        )
        if resp.status_code == 200:
            fields = extract_openalex_fields(resp.json())
            if fields:
                logger.debug("OpenAlex DOI lookup found %d fields for %s", len(fields), doi)
                return fields
    except Exception:
        logger.debug("OpenAlex DOI lookup failed for %s", doi, exc_info=True)
    return None


def _lookup_openalex_by_title(title: str) -> Optional[List[str]]:
    """Search OpenAlex by title and verify with Jaccard similarity."""
    try:
        resp = retry_with_backoff(
            lambda: httpx.get(
                f"{OPENALEX_API}/works",
                params={"search": title, "per_page": 1, "mailto": _openalex_email()},
                headers={"User-Agent": "research-agent/1.0"},
                timeout=10,
            ),
            max_retries=2,
            base_delay=1.0,
            retry_on=(429, 503, 504),
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            if results:
                hit = results[0]
                hit_title = hit.get("title", "")
                if _titles_match(title, hit_title):
                    fields = extract_openalex_fields(hit)
                    if fields:
                        logger.debug(
                            "OpenAlex title search found %d fields for '%s'",
                            len(fields), title[:60],
                        )
                        return fields
    except Exception:
        logger.debug("OpenAlex title search failed for '%s'", title[:60], exc_info=True)
    return None


# ---------------------------------------------------------------------------
# LLM fallback
# ---------------------------------------------------------------------------


def _extract_fields_via_llm(title: str, abstract: Optional[str] = None) -> Optional[List[str]]:
    """Ask an LLM to extract 3-5 specific academic sub-fields from title+abstract.

    Tries Groq (free tier), then OpenAI as fallback.
    """
    text = title
    if abstract:
        text += f"\n\nAbstract: {abstract[:500]}"

    prompt = (
        "Given this academic paper, list 3-5 specific academic fields it belongs to. "
        "Be specific — e.g. 'Indigenous Rights' not 'Political Science', "
        "'Urban Geography' not 'Geography'.\n\n"
        f"Paper: {text}\n\n"
        "Reply with ONLY a JSON array of field names, e.g. [\"Field1\", \"Field2\", \"Field3\"]."
    )

    providers = [
        ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY", "llama-3.1-8b-instant"),
        ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY", "gpt-4o-mini"),
    ]

    for url, key_env, model in providers:
        api_key = os.environ.get(key_env)
        if not api_key:
            continue
        try:
            resp = httpx.post(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0,
                },
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            answer = resp.json()["choices"][0]["message"]["content"].strip()
            fields = json.loads(answer)
            if isinstance(fields, list) and all(isinstance(f, str) for f in fields):
                logger.debug("LLM extracted %d fields: %s", len(fields), fields)
                return fields[:5]
        except (json.JSONDecodeError, KeyError, IndexError):
            logger.debug("LLM field extraction returned unparseable response from %s", model)
        except Exception:
            logger.debug("LLM field extraction failed with %s", model, exc_info=True)

    return None


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def enrich_fields(
    fields: Optional[List[str]],
    doi: Optional[str] = None,
    title: Optional[str] = None,
    abstract: Optional[str] = None,
) -> List[str]:
    """Enrich paper fields if they are missing or coarse.

    Pipeline:
      1. Return existing fields if they are already specific enough.
      2. Try OpenAlex by DOI, then by title.
      3. Fall back to LLM extraction from title + abstract.
      4. Return original fields if everything fails.
    """
    if not fields_need_enrichment(fields):
        return fields  # type: ignore[return-value]

    # Try OpenAlex DOI lookup
    if doi:
        oa_fields = _lookup_openalex_by_doi(doi)
        if oa_fields:
            return oa_fields

    # Try OpenAlex title search
    if title:
        oa_fields = _lookup_openalex_by_title(title)
        if oa_fields:
            return oa_fields

    # LLM fallback
    if title:
        llm_fields = _extract_fields_via_llm(title, abstract)
        if llm_fields:
            return llm_fields

    return fields or []
