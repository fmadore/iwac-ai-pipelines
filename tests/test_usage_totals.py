"""Token and cost accounting on the shared LLM clients."""

from types import SimpleNamespace

from common.llm_provider import UsageTotals, _attr


def test_totals_accumulate_and_ignore_what_a_provider_does_not_report():
    usage = UsageTotals()
    usage.add(input_tokens=100, output_tokens=20, cost_usd=0.001)
    usage.add(input_tokens="50", output_tokens=None, reasoning_tokens=7)
    usage.add(input_tokens=True)  # a bool is not a count

    assert usage.requests == 3
    assert usage.input_tokens == 150
    assert usage.output_tokens == 20
    assert usage.reasoning_tokens == 7
    assert usage.cost_usd == 0.001


def test_cost_stays_unknown_until_a_provider_states_it():
    usage = UsageTotals()
    usage.add(input_tokens=1, output_tokens=1)
    assert usage.cost_usd is None
    assert "$" not in usage.summary()
    usage.add(cost_usd=0.5)
    assert usage.summary().endswith("$0.5000")


def test_attr_walks_objects_and_dicts_alike():
    response = SimpleNamespace(usage={"prompt_tokens": 12, "details": SimpleNamespace(cached=3)})
    assert _attr(response, "usage", "prompt_tokens") == 12
    assert _attr(response, "usage", "details", "cached") == 3
    assert _attr(response, "usage", "missing", "deeper") is None
    assert _attr(None, "usage") is None
