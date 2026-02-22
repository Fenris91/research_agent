#!/usr/bin/env python3
"""
Researcher Lookup CLI

Lookup researcher profiles from academic APIs.

Usage:
    python -m research_agent.scripts.lookup_researchers
    python -m research_agent.scripts.lookup_researchers --file path/to/names.txt
    python -m research_agent.scripts.lookup_researchers --names "David Harvey, Doreen Massey"
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from research_agent.tools.researcher_lookup import ResearcherLookup, ResearcherProfile
from research_agent.tools.researcher_file_parser import (
    parse_researchers_file,
    parse_researchers_text,
)
from research_agent.utils.config import load_config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def print_profile(profile: ResearcherProfile):
    """Print researcher profile in a readable format."""
    print(f"\n{'=' * 60}")
    print(f"  {profile.name}")
    print(f"{'=' * 60}")

    if profile.affiliations:
        print(f"  Affiliations: {', '.join(profile.affiliations)}")

    print(f"  Works: {profile.works_count}")
    print(f"  Citations: {profile.citations_count:,}")

    if profile.h_index:
        print(f"  H-Index: {profile.h_index}")

    if profile.fields:
        print(f"  Fields: {', '.join(profile.fields[:5])}")

    if profile.openalex_id:
        print(f"  OpenAlex: https://openalex.org/{profile.openalex_id}")

    if profile.semantic_scholar_id:
        print(
            f"  S2: https://www.semanticscholar.org/author/{profile.semantic_scholar_id}"
        )

    if profile.web_results:
        print("\n  Web Results:")
        for i, r in enumerate(profile.web_results[:3], 1):
            print(f"    {i}. {r['title'][:50]}...")
            print(f"       {r['url']}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Lookup researcher profiles from academic APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default researchers.txt
  python -m research_agent.scripts.lookup_researchers

  # Use custom file
  python -m research_agent.scripts.lookup_researchers --file my_researchers.txt

  # Lookup specific names
  python -m research_agent.scripts.lookup_researchers --names "David Harvey, Doreen Massey"

  # Save output to specific directory
  python -m research_agent.scripts.lookup_researchers --output ./results/

  # Disable web search (faster)
  python -m research_agent.scripts.lookup_researchers --no-web
        """,
    )

    parser.add_argument(
        "--file",
        "-f",
        type=Path,
        help="Path to file with researcher names (default: data/researchers.txt)",
    )

    parser.add_argument(
        "--names", "-n", type=str, help="Comma-separated list of researcher names"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory for JSON files (default: data/researchers/)",
    )

    parser.add_argument(
        "--no-openalex", action="store_true", help="Disable OpenAlex lookup"
    )

    parser.add_argument(
        "--no-semantic-scholar",
        action="store_true",
        help="Disable Semantic Scholar lookup",
    )

    parser.add_argument("--no-web", action="store_true", help="Disable web search")

    parser.add_argument(
        "--email", type=str, help="Email for OpenAlex polite pool (optional)"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )

    parser.add_argument(
        "--json", action="store_true", help="Output results as JSON to stdout"
    )

    args = parser.parse_args()

    config = load_config()
    researcher_config = config.get("researcher_lookup", {})

    # Determine input source
    names = []

    if args.names:
        # From command line
        names = parse_researchers_text(args.names)
    elif args.file:
        # From specified file
        names = parse_researchers_file(args.file)
    else:
        # From default config location
        default_file = Path(
            researcher_config.get("input_file", "./data/researchers.txt")
        )
        if default_file.exists():
            names = parse_researchers_file(default_file)
        else:
            print(f"Error: No input file found at {default_file}")
            print("Use --file to specify a file or --names to provide names directly")
            sys.exit(1)

    if not names:
        print("No researcher names found in input")
        sys.exit(1)

    print(f"Looking up {len(names)} researchers...")

    # Determine output directory
    output_dir = args.output or Path(
        researcher_config.get("output_dir", "./data/researchers")
    )

    # Get email from config or args
    email = args.email or config.get("search", {}).get("openalex", {}).get("email")

    # Create lookup instance
    lookup = ResearcherLookup(
        email=email,
        request_delay=args.delay,
        use_openalex=not args.no_openalex,
        use_semantic_scholar=not args.no_semantic_scholar,
        use_web_search=not args.no_web,
    )

    try:
        # Progress callback
        def progress(current, total, name):
            if current < total:
                print(f"[{current + 1}/{total}] Looking up: {name}")

        # Run batch lookup
        profiles = await lookup.lookup_batch(
            names, output_dir=output_dir, progress_callback=progress
        )

        # Output results
        if args.json:
            # JSON output
            output = [p.to_dict() for p in profiles]
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            # Human-readable output
            for profile in profiles:
                print_profile(profile)

            # Save summary CSV
            csv_path = output_dir / "summary.csv"
            ResearcherLookup.save_summary_csv(profiles, csv_path)
            print(f"\n{'=' * 60}")
            print(f"Results saved to: {output_dir}")
            print(f"Summary CSV: {csv_path}")

    finally:
        await lookup.close()


if __name__ == "__main__":
    asyncio.run(main())
