"""
run_nb01.py — Literature Search Pipeline
NB-01: Busca bibliográfica para o artigo RBFin Offline RL Portfolio Optimization

Queries cobrem: Offline RL, RL in Finance, CVaR Risk, Regime Switching,
Benchmarks e mercado Brasileiro.

Outputs:
    - data/external/literature_db.csv
    - latex/references.bib
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_EXTERNAL = PROJECT_ROOT / "data" / "external"
LATEX_DIR = PROJECT_ROOT / "latex"
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_EXTERNAL / "literature_db.csv"
BIB_PATH = LATEX_DIR / "references.bib"

# ---------------------------------------------------------------------------
# Load .env (optional)
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

SS_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
CROSSREF_EMAIL = os.getenv("CROSSREF_EMAIL", "gustavo@ufpr.br")

# ---------------------------------------------------------------------------
# Search queries
# ---------------------------------------------------------------------------
QUERIES: list[tuple[str, str]] = [
    # Group 1: Offline RL
    ("offline reinforcement learning portfolio optimization", "offline_rl"),
    ("conservative Q-learning CQL finance", "offline_rl"),
    ("batch reinforcement learning financial", "offline_rl"),
    # Group 2: RL in Finance
    ("deep reinforcement learning portfolio management", "rl_finance"),
    ("reinforcement learning asset allocation", "rl_finance"),
    ("DDPG portfolio optimization", "rl_finance"),
    # Group 3: CVaR Risk
    ("conditional value at risk portfolio optimization", "cvar_risk"),
    ("CVaR reinforcement learning reward", "cvar_risk"),
    ("risk-sensitive reinforcement learning", "cvar_risk"),
    # Group 4: Regimes
    ("regime switching portfolio optimization", "regimes"),
    ("macroeconomic regime portfolio machine learning", "regimes"),
    ("emerging market portfolio reinforcement learning", "regimes"),
    # Group 5: Benchmarks
    ("mean variance portfolio optimization out-of-sample", "benchmarks"),
    ("risk parity portfolio DeMiguel", "benchmarks"),
    ("naive diversification 1/N portfolio", "benchmarks"),
    # Group 6: Brazil
    ("Brazilian stock market portfolio optimization", "brazil"),
    ("SELIC interest rate asset pricing Brazil", "brazil"),
]

# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------
COLUMNS = [
    "id", "title", "authors", "year", "journal", "doi", "url", "pdf_url",
    "citation_abnt", "bibtex_key", "bibtex_entry", "abstract", "keywords",
    "citation_count", "source", "validated", "relevance_tag", "notes", "added_at",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def reconstruct_abstract(inv_idx: dict | None) -> str:
    """Reconstruct OpenAlex abstract from inverted index."""
    if not inv_idx:
        return ""
    position_word: dict[int, str] = {}
    for word, positions in inv_idx.items():
        for pos in positions:
            position_word[pos] = word
    if not position_word:
        return ""
    return " ".join(position_word[i] for i in sorted(position_word))


def make_bibtex_key(authors_str: str, year: str | int) -> str:
    """Generate BibTeX key from first author last name + year."""
    year_str = str(year) if year else "0000"
    if not authors_str:
        return f"Unknown{year_str}"
    first_author = authors_str.split(";")[0].split(",")[0].split()[-1]
    # Remove non-alphanumeric
    first_author = re.sub(r"[^A-Za-z]", "", first_author)
    return f"{first_author}{year_str}" if first_author else f"Unknown{year_str}"


def make_abnt(authors_str: str, title: str, journal: str, year: str | int) -> str:
    """Generate ABNT-style citation."""
    year_str = str(year) if year else "s.d."
    authors_list = [a.strip() for a in authors_str.split(";") if a.strip()]
    if not authors_list:
        author_part = "AUTOR DESCONHECIDO"
    elif len(authors_list) == 1:
        parts = authors_list[0].split()
        if len(parts) >= 2:
            last = parts[-1].upper()
            first = " ".join(parts[:-1])
            author_part = f"{last}, {first}"
        else:
            author_part = authors_list[0].upper()
    else:
        parts = authors_list[0].split()
        if len(parts) >= 2:
            last = parts[-1].upper()
            first = " ".join(parts[:-1])
            author_part = f"{last}, {first} et al."
        else:
            author_part = authors_list[0].upper() + " et al."
    journal_part = journal if journal else "s.l."
    return f"{author_part}. {title}. {journal_part}, {year_str}."


def make_bibtex(key: str, title: str, authors_str: str, year: str | int,
                journal: str, doi: str, abstract: str) -> str:
    """Generate BibTeX @article entry."""
    year_str = str(year) if year else ""
    authors_bibtex = " and ".join(a.strip() for a in authors_str.split(";") if a.strip())
    doi_field = f"  doi = {{{doi}}},\n" if doi else ""
    abstract_short = abstract[:400].replace("{", "").replace("}", "") if abstract else ""
    abstract_field = f"  abstract = {{{abstract_short}}},\n" if abstract_short else ""
    entry_type = "article" if journal else "misc"
    return (
        f"@{entry_type}{{{key},\n"
        f"  title = {{{{{title}}}}},\n"
        f"  author = {{{authors_bibtex}}},\n"
        f"  year = {{{year_str}}},\n"
        f"  journal = {{{journal}}},\n"
        f"{doi_field}"
        f"{abstract_field}"
        f"}}\n"
    )


def classify_relevance(title: str, abstract: str) -> str:
    """Assign relevance_tag based on keyword matching."""
    text = (title + " " + abstract).lower()
    core_kws = ["offline rl", "offline reinforcement", "conservative q", "cql",
                "cvar reward", "portfolio rl", "conservative q-learning"]
    supporting_kws = ["portfolio optimization", "reinforcement learning", "risk parity",
                      "cvar", "conditional value at risk", "sharpe", "asset allocation",
                      "regime switching", "portfolio management"]
    for kw in core_kws:
        if kw in text:
            return "core"
    for kw in supporting_kws:
        if kw in text:
            return "supporting"
    return "context"


# ---------------------------------------------------------------------------
# Semantic Scholar search
# ---------------------------------------------------------------------------

def search_semantic_scholar(query: str, limit: int = 20) -> list[dict[str, Any]]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params: dict[str, Any] = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,venue,citationCount,externalIds,abstract",
    }
    headers = {}
    if SS_API_KEY and SS_API_KEY != "your_key_here":
        headers["x-api-key"] = SS_API_KEY
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json()
        papers = data.get("data", [])
        results = []
        for p in papers:
            doi = (p.get("externalIds") or {}).get("DOI", "")
            authors_list = [a.get("name", "") for a in (p.get("authors") or [])]
            results.append({
                "title": clean_text(p.get("title", "")),
                "authors": "; ".join(authors_list),
                "year": p.get("year") or "",
                "journal": clean_text(p.get("venue", "")),
                "doi": doi,
                "url": f"https://www.semanticscholar.org/paper/{p.get('paperId', '')}",
                "pdf_url": "",
                "abstract": clean_text(p.get("abstract", "")),
                "citation_count": p.get("citationCount", 0) or 0,
                "source": "semantic_scholar",
            })
        return results
    except Exception as exc:
        print(f"    [SS ERROR] {exc}")
        return []


# ---------------------------------------------------------------------------
# OpenAlex search
# ---------------------------------------------------------------------------

def search_openalex(query: str, per_page: int = 10) -> list[dict[str, Any]]:
    url = "https://api.openalex.org/works"
    params: dict[str, Any] = {
        "search": query,
        "per-page": per_page,
        "select": (
            "title,authorships,publication_year,primary_location,"
            "cited_by_count,doi,abstract_inverted_index,best_oa_location"
        ),
    }
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        works = data.get("results", [])
        results = []
        for w in works:
            doi_raw = w.get("doi", "") or ""
            doi = doi_raw.replace("https://doi.org/", "").strip()
            authors_list = [
                a.get("author", {}).get("display_name", "")
                for a in (w.get("authorships") or [])
            ]
            journal = ""
            loc = w.get("primary_location") or {}
            source = loc.get("source") or {}
            if source:
                journal = clean_text(source.get("display_name", ""))
            pdf_url = ""
            oa = w.get("best_oa_location") or {}
            if oa:
                pdf_url = oa.get("pdf_url", "") or ""
            abstract = reconstruct_abstract(w.get("abstract_inverted_index"))
            results.append({
                "title": clean_text(w.get("title", "")),
                "authors": "; ".join(authors_list),
                "year": w.get("publication_year") or "",
                "journal": journal,
                "doi": doi,
                "url": doi_raw if doi_raw else "",
                "pdf_url": pdf_url,
                "abstract": clean_text(abstract),
                "citation_count": w.get("cited_by_count", 0) or 0,
                "source": "openalex",
            })
        return results
    except Exception as exc:
        print(f"    [OA ERROR] {exc}")
        return []


# ---------------------------------------------------------------------------
# Crossref validation
# ---------------------------------------------------------------------------

def validate_crossref(doi: str) -> tuple[bool, dict[str, Any]]:
    """Validate a DOI via Crossref and return (is_valid, metadata)."""
    if not doi:
        return False, {}
    url = f"https://api.crossref.org/works/{doi}"
    params = {"mailto": CROSSREF_EMAIL}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            msg = r.json().get("message", {})
            return True, msg
        return False, {}
    except Exception:
        return False, {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    print("=" * 60)
    print("RBFin NB-01 — Literature Search Pipeline")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_papers: list[dict[str, Any]] = []
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    paper_id_counter = 0

    total_queries = len(QUERIES)

    for idx, (query, group) in enumerate(QUERIES, start=1):
        print(f"\nQuery {idx}/{total_queries}: \"{query}\" [{group}]")

        # --- Semantic Scholar ---
        print("  -> Semantic Scholar...", end=" ", flush=True)
        ss_results = search_semantic_scholar(query)
        print(f"{len(ss_results)} results")
        time.sleep(1.2)

        # --- OpenAlex ---
        print("  -> OpenAlex...", end=" ", flush=True)
        oa_results = search_openalex(query)
        print(f"{len(oa_results)} results")
        time.sleep(0.6)

        combined = ss_results + oa_results
        added = 0
        for p in combined:
            title_key = p["title"].lower()[:80]
            doi = p.get("doi", "").strip().lower()
            # Deduplicate
            if doi and doi in seen_dois:
                continue
            if title_key in seen_titles:
                continue
            if not p["title"]:
                continue

            if doi:
                seen_dois.add(doi)
            seen_titles.add(title_key)

            paper_id_counter += 1
            p["id"] = f"P{paper_id_counter:04d}"
            p["query_group"] = group
            all_papers.append(p)
            added += 1

        print(f"  -> Added {added} new unique papers (total so far: {len(all_papers)})")

    print(f"\n{'=' * 60}")
    print(f"Total unique papers collected: {len(all_papers)}")
    print("Starting Crossref validation (papers with DOI)...")

    # ---------------------------------------------------------------------------
    # Crossref validation
    # ---------------------------------------------------------------------------
    validated_count = 0
    papers_with_doi = [p for p in all_papers if p.get("doi")]
    print(f"Papers with DOI to validate: {len(papers_with_doi)}")

    doi_to_crossref: dict[str, dict] = {}
    for i, p in enumerate(papers_with_doi, start=1):
        doi = p["doi"]
        if i % 10 == 0:
            print(f"  Validating {i}/{len(papers_with_doi)}...")
        is_valid, cf_data = validate_crossref(doi)
        if is_valid:
            doi_to_crossref[doi.lower()] = cf_data
            validated_count += 1
        time.sleep(0.3)

    print(f"Validated: {validated_count}/{len(papers_with_doi)}")

    # ---------------------------------------------------------------------------
    # Enrich and build final records
    # ---------------------------------------------------------------------------
    rows: list[dict[str, Any]] = []
    bibtex_keys_used: set[str] = set()
    bibtex_entries: list[str] = []
    now_str = datetime.now(timezone.utc).isoformat()

    for p in all_papers:
        doi = (p.get("doi") or "").strip()
        doi_lower = doi.lower()

        # Try to enrich from Crossref
        cf = doi_to_crossref.get(doi_lower, {})
        if cf:
            # Update journal if better data available
            if not p.get("journal"):
                container = cf.get("container-title", [])
                if container:
                    p["journal"] = container[0]
            # Update year if missing
            if not p.get("year"):
                pub = cf.get("published-print") or cf.get("published-online") or {}
                dp = pub.get("date-parts", [[]])
                if dp and dp[0]:
                    p["year"] = dp[0][0]

        validated = doi_lower in doi_to_crossref

        # Generate BibTeX key (ensure uniqueness)
        base_key = make_bibtex_key(p.get("authors", ""), p.get("year", ""))
        key = base_key
        suffix = 97  # 'a'
        while key in bibtex_keys_used:
            key = base_key + chr(suffix)
            suffix += 1
        bibtex_keys_used.add(key)

        title = p.get("title", "")
        authors_str = p.get("authors", "")
        year = p.get("year", "")
        journal = p.get("journal", "")
        abstract = p.get("abstract", "")

        abnt = make_abnt(authors_str, title, journal, year)
        bibtex = make_bibtex(key, title, authors_str, year, journal, doi, abstract)
        bibtex_entries.append(bibtex)

        relevance = classify_relevance(title, abstract)

        rows.append({
            "id": p.get("id", ""),
            "title": title,
            "authors": authors_str,
            "year": year,
            "journal": journal,
            "doi": doi,
            "url": p.get("url", ""),
            "pdf_url": p.get("pdf_url", ""),
            "citation_abnt": abnt,
            "bibtex_key": key,
            "bibtex_entry": bibtex.replace("\n", " "),
            "abstract": abstract,
            "keywords": p.get("query_group", ""),
            "citation_count": p.get("citation_count", 0),
            "source": p.get("source", ""),
            "validated": validated,
            "relevance_tag": relevance,
            "notes": "",
            "added_at": now_str,
        })

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved CSV: {CSV_PATH} ({len(rows)} rows)")

    # ---------------------------------------------------------------------------
    # Save BibTeX
    # ---------------------------------------------------------------------------
    bib_header = (
        "% references.bib — Auto-generated by run_nb01.py\n"
        f"% Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"% Total entries: {len(bibtex_entries)}\n\n"
    )
    with open(BIB_PATH, "w", encoding="utf-8") as f:
        f.write(bib_header)
        for entry in bibtex_entries:
            f.write(entry + "\n")
    print(f"Saved BibTeX: {BIB_PATH} ({len(bibtex_entries)} entries)")

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    core = sum(1 for r in rows if r["relevance_tag"] == "core")
    supporting = sum(1 for r in rows if r["relevance_tag"] == "supporting")
    context = sum(1 for r in rows if r["relevance_tag"] == "context")
    with_doi = sum(1 for r in rows if r["doi"])

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total papers:      {len(rows)}")
    print(f"  core:            {core}")
    print(f"  supporting:      {supporting}")
    print(f"  context:         {context}")
    print(f"Papers with DOI:   {with_doi}")
    print(f"Validated (Crossref): {validated_count}")
    print(f"Output CSV:        {CSV_PATH}")
    print(f"Output BibTeX:     {BIB_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
