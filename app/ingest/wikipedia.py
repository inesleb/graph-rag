"""Download articles from French Wikipedia.

Reads a list of articles (title, kind) and saves one JSON file per
article under data/raw/, with the plain text, the revision it was taken from
and the links to other articles of the corpus.

- "affair" articles keep their full text, minus the reference sections at the end.
- "context" articles (people, parties, institutions) keep only their intro, so
  that very long biographies do not swamp the corpus.

Wikipedia text is CC BY-SA 4.0: every file keeps the article URL and revision
id so the source can be credited.
"""

import csv
import json
import os
import re
import time
from pathlib import Path

import requests

API_URL = "https://fr.wikipedia.org/w/api.php"
USER_AGENT = os.getenv("WIKI_USER_AGENT", "graph-rag-class-project/0.1 (university course)")
MAX_RETRIES = 5
REQUEST_DELAY = 0.5  # seconds between requests, to stay under Wikipedia's rate limit
# Sections after which an article is only references and navigation.
END_SECTIONS = re.compile(
    r"^==\s*(Notes et références|Références|Voir aussi|Bibliographie|"
    r"Liens externes|Articles connexes|Annexes)\s*==",
    re.MULTILINE,
)


def get_json(session: requests.Session, params: dict) -> dict:
    """Send one API query, waiting and retrying when Wikipedia rate-limits us."""
    params = {"action": "query", "format": "json", "formatversion": 2, **params}
    for attempt in range(MAX_RETRIES):
        response = session.get(API_URL, params=params, timeout=30)
        if response.status_code != 429:
            break
        time.sleep(float(response.headers.get("Retry-After", 2**attempt)))
    response.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return response.json()


def clean_text(text: str, max_chars: int) -> str:
    """Drop the reference sections at the end and cap the length."""
    match = END_SECTIONS.search(text)
    if match:
        text = text[: match.start()]
    return text.strip()[:max_chars]


def fetch_article(session: requests.Session, title: str, intro_only: bool) -> dict:
    """Fetch the plain text and revision metadata of one article."""
    params = {
        "titles": title,
        "redirects": 1,
        "prop": "extracts|info|revisions",
        "explaintext": 1,
        "inprop": "url",
        "rvprop": "ids|timestamp",
    }
    if intro_only:
        params["exintro"] = 1
    page = get_json(session, params)["query"]["pages"][0]
    if page.get("missing"):
        raise ValueError(f"No Wikipedia article named {title!r}")
    revision = page["revisions"][0]
    return {
        "pageid": page["pageid"],
        "title": page["title"],
        "url": page["fullurl"],
        "revid": revision["revid"],
        "timestamp": revision["timestamp"],
        "text": page.get("extract", ""),
    }


def fetch_links(session: requests.Session, title: str) -> set[str]:
    """Return the titles an article links to, with redirects resolved."""
    params = {
        "titles": title,
        "generator": "links",
        "gplnamespace": 0,
        "gpllimit": "max",
        "redirects": 1,
    }
    links = set()
    while True:
        data = get_json(session, params)
        links.update(page["title"] for page in data.get("query", {}).get("pages", []))
        if "continue" not in data:
            return links
        params = {**params, **data["continue"]}


def read_article_list(path: Path) -> list[dict]:
    """Read the list of articles to download (columns: title, kind)."""
    with path.open(encoding="utf-8") as file:
        return list(csv.DictReader(file))


def new_session() -> requests.Session:
    """A requests session identifying the project to Wikipedia."""
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def download(
    entries: list[dict],
    out: Path,
    session: requests.Session,
    max_chars: int = 40_000,
    known_titles: set[str] | None = None,
) -> list[dict]:
    """Download the listed articles (dicts with title and kind) into `out`; return the index.

    Links are kept when they point to another downloaded article or to `known_titles`.
    """
    articles = []
    for entry in entries:
        article = fetch_article(session, entry["title"], intro_only=entry["kind"] == "context")
        article["kind"] = entry["kind"]
        article["text"] = clean_text(article["text"], max_chars)
        article["links"] = fetch_links(session, article["title"])
        articles.append(article)
        print(f"  {article['kind']:7} {len(article['text']):>6} chars  {article['title']}")

    corpus_titles = {article["title"] for article in articles} | (known_titles or set())
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for article in articles:
        article["links"] = sorted(article["links"] & corpus_titles - {article["title"]})
        path = out / f"{article['pageid']}.json"
        path.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        index.append(
            {key: article[key] for key in ("pageid", "title", "kind", "url", "revid")}
            | {"chars": len(article["text"]), "links": len(article["links"])}
        )
    return index
