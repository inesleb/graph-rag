"""Test doubles for the LLM and the Neo4j driver."""


class FakeStructured:
    """What llm.with_structured_output(...) returns: answers from a queue per schema."""

    def __init__(self, replies):
        self.replies = replies
        self.prompts = []

    def invoke(self, prompt):
        """Return the next canned reply."""
        self.prompts.append(prompt)
        return self.replies.pop(0)


class FakeLLM:
    """A chat model whose structured outputs are canned, keyed by schema name."""

    def __init__(self, **replies_by_schema):
        self.structured = {
            name: FakeStructured(list(replies)) for name, replies in replies_by_schema.items()
        }

    def with_structured_output(self, schema, method=None):
        """Return the canned responder for this schema."""
        assert method
        return self.structured[schema.__name__]


class FakeRecord(dict):
    """A neo4j Record stand-in (dict access is all the code uses)."""


class FakeResult:
    """What driver.execute_query returns."""

    def __init__(self, records):
        self.records = [FakeRecord(r) for r in records]


class FakeDriver:
    """Returns canned records, chosen by a substring of the query."""

    def __init__(self, answers):
        self.answers = answers
        self.queries = []

    def execute_query(self, query_, **params):
        """Return the records of the first matching answer."""
        self.queries.append((query_, params))
        for marker, records in self.answers.items():
            if marker in query_:
                return FakeResult(records(params) if callable(records) else records)
        return FakeResult([])
