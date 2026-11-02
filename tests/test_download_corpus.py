"""Tests for the corpus download script, with the Wikipedia API mocked."""

import json

import pytest

from scripts import download_corpus as dc

ARTICLE_TEXT = "Résumé.\n\n== Faits ==\nDes faits.\n\n== Notes et références ==\nRéf."


@pytest.fixture(autouse=True)
def no_delay(monkeypatch):
    """Do not wait between requests during tests."""
    monkeypatch.setattr(dc, "REQUEST_DELAY", 0)


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.headers = {"Retry-After": "0"}

    def raise_for_status(self):
        """Never fails."""

    def json(self):
        """Return the canned payload."""
        return self.payload


class FakeSession:
    """Answers MediaWiki queries from canned data."""

    pages = {"Affaire A": 1, "Personne B": 2}
    links = {"Affaire A": ["Personne B", "Paris"], "Personne B": ["Affaire A"]}

    def __init__(self):
        self.headers = {}
        self.calls = []

    def get(self, _url, params, timeout):
        """Dispatch on the query type."""
        assert timeout
        self.calls.append(params)
        title = params["titles"]
        if title not in self.pages:
            return FakeResponse({"query": {"pages": [{"title": title, "missing": True}]}})
        if params.get("generator") == "links":
            pages = [{"title": link} for link in self.links[title]]
            return FakeResponse({"query": {"pages": pages}})
        page = {
            "pageid": self.pages[title],
            "title": title,
            "fullurl": f"https://fr.wikipedia.org/wiki/{title}",
            "revisions": [{"revid": 100, "timestamp": "2026-01-01T00:00:00Z"}],
            "extract": ARTICLE_TEXT,
        }
        return FakeResponse({"query": {"pages": [page]}})


def write_list(tmp_path, rows):
    """Write an articles.csv and return its path."""
    path = tmp_path / "articles.csv"
    path.write_text("title,kind\n" + "".join(f"{t},{k}\n" for t, k in rows), encoding="utf-8")
    return path


def test_clean_text_drops_reference_sections():
    """Everything from 'Notes et références' on is removed."""
    assert dc.clean_text(ARTICLE_TEXT, 1000) == "Résumé.\n\n== Faits ==\nDes faits."


def test_clean_text_caps_length():
    """Text is cut at max_chars."""
    assert dc.clean_text("abcdef", 3) == "abc"


def test_get_json_retries_after_rate_limit():
    """A 429 answer is retried instead of failing the download."""

    class Limited:
        """Rate-limits the first call."""

        calls = 0

        def get(self, _url, params, timeout):
            """Answer 429 once, then succeed."""
            assert timeout and params
            self.calls += 1
            return FakeResponse({"ok": True}, status_code=429 if self.calls == 1 else 200)

    session = Limited()
    assert dc.get_json(session, {}) == {"ok": True}
    assert session.calls == 2


def test_fetch_article_requests_intro_only_for_context():
    """Context pages are fetched with exintro, affairs without."""
    session = FakeSession()
    dc.fetch_article(session, "Personne B", intro_only=True)
    dc.fetch_article(session, "Affaire A", intro_only=False)
    assert session.calls[0]["exintro"] == 1
    assert "exintro" not in session.calls[1]


def test_fetch_article_rejects_missing_title():
    """A title that does not exist raises a clear error."""
    with pytest.raises(ValueError, match="Inconnue"):
        dc.fetch_article(FakeSession(), "Inconnue", intro_only=False)


def test_fetch_links_follows_continuation():
    """Every continuation page of links is requested."""

    class Paged(FakeSession):
        """Returns links over two result pages."""

        def get(self, _url, params, timeout):
            """Continue once."""
            self.calls.append(params)
            if "gplcontinue" in params:
                return FakeResponse({"query": {"pages": [{"title": "B"}]}})
            return FakeResponse(
                {"query": {"pages": [{"title": "A"}]}, "continue": {"gplcontinue": "x"}}
            )

    assert dc.fetch_links(Paged(), "Affaire A") == {"A", "B"}


def test_main_saves_articles_and_index(tmp_path, monkeypatch, capsys):
    """Files are written, links stay inside the corpus, the index is complete."""
    monkeypatch.setattr(dc.requests, "Session", FakeSession)
    articles = write_list(tmp_path, [("Affaire A", "affair"), ("Personne B", "context")])
    dc.main(["--articles", str(articles), "--out", str(tmp_path / "raw")])

    saved = json.loads((tmp_path / "raw" / "1.json").read_text(encoding="utf-8"))
    assert saved["links"] == ["Personne B"]
    assert saved["kind"] == "affair"
    assert "Réf." not in saved["text"]
    index = json.loads((tmp_path / "corpus_index.json").read_text(encoding="utf-8"))
    assert [entry["title"] for entry in index] == ["Affaire A", "Personne B"]
    assert "Saved 2 articles" in capsys.readouterr().out
