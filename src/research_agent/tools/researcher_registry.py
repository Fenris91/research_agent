"""
Researcher Registry - In-memory store for looked-up researchers.

This module provides a singleton registry that stores ResearcherProfile objects
looked up in the Researcher Lookup tab, making them available for selection
in the Citation Explorer tab.
"""

import logging
from typing import Dict, List, Optional
from threading import Lock

from research_agent.tools.researcher_lookup import ResearcherProfile, AuthorPaper

logger = logging.getLogger(__name__)


class ResearcherRegistry:
    """
    Singleton registry for storing and retrieving researcher profiles.

    Thread-safe storage for researcher profiles that can be accessed
    across different UI tabs.

    Usage:
        registry = ResearcherRegistry.get_instance()
        registry.add(profile)
        researcher = registry.get("David Harvey")
        all_researchers = registry.list_all()
    """

    _instance: Optional["ResearcherRegistry"] = None
    _lock = Lock()

    def __new__(cls) -> "ResearcherRegistry":
        """Ensure singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._researchers: Dict[str, ResearcherProfile] = {}
                cls._instance._registry_lock = Lock()
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ResearcherRegistry":
        """Get the singleton registry instance."""
        return cls()

    def add(self, profile: ResearcherProfile) -> None:
        """
        Add or update a researcher profile in the registry.

        Args:
            profile: ResearcherProfile to store
        """
        with self._registry_lock:
            key = profile.normalized_name or profile.name.lower().strip()
            self._researchers[key] = profile
            logger.debug(f"Added researcher to registry: {profile.name}")

    def add_batch(self, profiles: List[ResearcherProfile]) -> None:
        """
        Add multiple researcher profiles to the registry.

        Args:
            profiles: List of ResearcherProfile objects
        """
        with self._registry_lock:
            for profile in profiles:
                key = profile.normalized_name or profile.name.lower().strip()
                self._researchers[key] = profile
            logger.info(f"Added {len(profiles)} researchers to registry")

    def get(self, name: str) -> Optional[ResearcherProfile]:
        """
        Get a researcher profile by name.

        Args:
            name: Researcher name (case-insensitive)

        Returns:
            ResearcherProfile or None if not found
        """
        with self._registry_lock:
            key = name.lower().strip()
            return self._researchers.get(key)

    def get_by_id(
        self, openalex_id: Optional[str] = None, s2_id: Optional[str] = None
    ) -> Optional[ResearcherProfile]:
        """
        Get a researcher profile by OpenAlex or Semantic Scholar ID.

        Args:
            openalex_id: OpenAlex author ID
            s2_id: Semantic Scholar author ID

        Returns:
            ResearcherProfile or None if not found
        """
        with self._registry_lock:
            for profile in self._researchers.values():
                if openalex_id and profile.openalex_id == openalex_id:
                    return profile
                if s2_id and profile.semantic_scholar_id == s2_id:
                    return profile
            return None

    def list_all(self) -> List[ResearcherProfile]:
        """
        Get all researcher profiles in the registry.

        Returns:
            List of all stored ResearcherProfile objects
        """
        with self._registry_lock:
            return list(self._researchers.values())

    def list_names(self) -> List[str]:
        """
        Get names of all researchers in the registry.

        Returns:
            List of researcher names
        """
        with self._registry_lock:
            return [p.name for p in self._researchers.values()]

    def list_with_papers(self) -> List[ResearcherProfile]:
        """
        Get all researchers that have papers fetched.

        Returns:
            List of ResearcherProfile objects that have top_papers
        """
        with self._registry_lock:
            return [p for p in self._researchers.values() if p.top_papers]

    def remove(self, name: str) -> bool:
        """
        Remove a researcher from the registry.

        Args:
            name: Researcher name to remove

        Returns:
            True if removed, False if not found
        """
        with self._registry_lock:
            key = name.lower().strip()
            if key in self._researchers:
                del self._researchers[key]
                logger.debug(f"Removed researcher from registry: {name}")
                return True
            return False

    def clear(self) -> None:
        """Clear all researchers from the registry."""
        with self._registry_lock:
            self._researchers.clear()
            logger.info("Cleared researcher registry")

    def count(self) -> int:
        """Get the number of researchers in the registry."""
        with self._registry_lock:
            return len(self._researchers)

    def get_all_papers(self) -> List[AuthorPaper]:
        """
        Get all papers from all researchers in the registry.

        Returns:
            List of all AuthorPaper objects, deduplicated by paper_id
        """
        with self._registry_lock:
            papers_by_id: Dict[str, AuthorPaper] = {}
            for profile in self._researchers.values():
                for paper in profile.top_papers:
                    if paper.paper_id and paper.paper_id not in papers_by_id:
                        papers_by_id[paper.paper_id] = paper
            return list(papers_by_id.values())

    def to_dropdown_choices(self) -> List[tuple]:
        """
        Get researcher choices for a Gradio dropdown.

        Returns:
            List of (display_name, normalized_name) tuples
        """
        with self._registry_lock:
            choices = []
            for profile in self._researchers.values():
                # Format: "Name (N papers, M citations)"
                papers_count = len(profile.top_papers)
                display = f"{profile.name} ({papers_count} papers, {profile.citations_count:,} citations)"
                choices.append((display, profile.normalized_name or profile.name.lower()))
            return sorted(choices, key=lambda x: x[0])


# Convenience function for getting the singleton
def get_researcher_registry() -> ResearcherRegistry:
    """Get the global researcher registry instance."""
    return ResearcherRegistry.get_instance()


__all__ = [
    "ResearcherRegistry",
    "get_researcher_registry",
]
