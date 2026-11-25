"""Offline tests: no API keys, no network."""

import json
import socket
from types import SimpleNamespace

import pytest
from crewai.tasks.task_output import TaskOutput

from procurement.config import COMPACT, Company, ConfigError, ProcurementRequest, Weights, missing_keys
from procurement.crew import run_procurement
from procurement.demo import (
    DEMO_REQUEST,
    DemoScrapeGraph,
    DemoTavily,
    _answer,
    demo_script,
    demo_services,
)
from procurement.guardrails import Guardrails
from procurement.ranking import rank_products
from procurement.report import render_report
from procurement.schemas import (
    Assessment,
    Product,
    ReportContent,
    Requirement,
    RequirementCheck,
)
from procurement.state import (
    RunState,
    ScrapedPage,
    normalise_url,
    numbers_in,
    spread_over_websites,
)
from procurement.tools import ScrapeGraphTool, TavilySearchTool


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    real_connect = socket.socket.connect

    def guarded(self, address):
        # asyncio on Windows needs loopback sockets
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
            raise AssertionError(f"Network access in a test: {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


@pytest.fixture
def company():
    return Company.load()


@pytest.fixture
def state(company):
    return RunState(company=company, request=DEMO_REQUEST)


def output(data):
    raw = data if isinstance(data, str) else json.dumps(data)
    return TaskOutput(description="t", raw=raw, agent="a")


def product(pid="P1", price=100.0, currency="EGP", in_stock=True, url=None):
    return Product(
        id=pid, title=f"Item {pid}", url=url or f"https://{pid.lower()}.example/x", price=price,
        currency=currency, in_stock=in_stock, seller=None, warranty=None, specs=[],
    )


def assessment(pid, **statuses):
    return Assessment(
        product_id=pid, pros=[], cons=[],
        checks=[RequirementCheck(requirement_id=k, status=v, evidence="") for k, v in statuses.items()],
    )


def add_page(state, url, text):
    key = normalise_url(url)
    state.found_urls[key] = url
    state.candidates.append(key)
    state.pages[key] = ScrapedPage(url=url, tool="ScrapeGraph", text=text)


REQS = [
    Requirement(id="R1", text="16 GB RAM", must_have=True),
    Requirement(id="R2", text="Backlight", must_have=False),
]


# --- whole crew ---------------------------------------------------------------
def test_full_crew_run(company):
    state, content = run_procurement(
        DEMO_REQUEST, company, demo_services(), demo=True, verbose=False
    )
    assert state.recommended.product.id == content.recommended_product_id == "P1"
    assert [r.product.id for r in state.ranking if r.eligible] == ["P1", "P2"]
    assert len(state.pages) == 5 and len(state.queries) == 3
    html = render_report(state, content)
    assert "Demo data." in html
    assert "385,000" in html  # 10 units at 38,500


def test_crew_retries_after_a_wrong_recommendation(company):
    script = demo_script()
    wrong = json.loads(script["report"][-1].split("Final Answer: ")[1])
    wrong["recommended_product_id"] = "P5"
    script["report"].insert(1, _answer(wrong))
    _, content = run_procurement(DEMO_REQUEST, company, demo_services(script), verbose=False)
    assert content.recommended_product_id == "P1"


def test_crew_drops_hallucinated_products_and_prices(company):
    script = demo_script()
    answer = json.loads(script["scrape"][-1].split("Final Answer: ")[1])
    answer["products"][1]["price"] = 1.0  # not on the page
    answer["products"].append({**answer["products"][0], "url": "https://fake.example/x"})
    script["scrape"][-1] = _answer(answer)
    state, _ = run_procurement(DEMO_REQUEST, company, demo_services(script), verbose=False)
    assert len(state.products) == 5
    assert next(p for p in state.products if p.id == "P2").price is None
    assert any("does not appear on its page" in w for w in state.warnings)
    assert any("was not scraped" in w for w in state.warnings)


def test_crew_stops_when_no_page_can_be_scraped(company):
    services = demo_services()
    services.scrapegraph = SimpleNamespace(
        extract=lambda *a, **k: SimpleNamespace(status="error", data=None, error="HTTP 403")
    )
    with pytest.raises(Exception, match="HTTP 403"):
        run_procurement(DEMO_REQUEST, company, services, verbose=False)


# --- ranking ------------------------------------------------------------------
def test_scores(company):
    ranked = rank_products(
        [product("P1", 100), product("P2", 200)],
        [assessment("P1", R1="met", R2="unmet"), assessment("P2", R1="met", R2="met")],
        REQS, ProcurementRequest(product="laptop"), company,
    )
    by_id = {r.product.id: r for r in ranked}
    assert by_id["P1"].price_score == 100 and by_id["P2"].price_score == 50
    assert by_id["P1"].spec_score == pytest.approx(200 / 3)
    assert by_id["P2"].spec_score == 100
    assert by_id["P1"].value_score == 100 and by_id["P2"].value_score == pytest.approx(75)
    assert by_id["P1"].total_score == pytest.approx(35 + 0.4 * 200 / 3 + 25)
    assert [r.rank for r in ranked] == [1, 2]


@pytest.mark.parametrize(
    "item, checks, issue",
    [
        (product(price=None), {"R1": "met"}, "No verified price"),
        (product(currency="USD"), {"R1": "met"}, "Priced in USD"),
        (product(currency=None), {"R1": "met"}, "Currency not stated"),
        (product(in_stock=False), {"R1": "met"}, "Out of stock"),
        (product(price=500), {"R1": "met"}, "Over budget"),
        (product(), {"R1": "unmet"}, "Must-have not met"),
    ],
)
def test_blocking_issues(company, item, checks, issue):
    request = ProcurementRequest(product="laptop", max_unit_price=300)
    (ranked,) = rank_products([item], [assessment("P1", **checks)], REQS, request, company)
    assert not ranked.eligible and ranked.rank is None
    assert issue in ranked.issues[0]


def test_unknown_must_have_needs_confirmation_but_does_not_block(company):
    (r,) = rank_products(
        [product()], [assessment("P1", R1="unknown")], REQS,
        ProcurementRequest(product="laptop"), company,
    )
    assert r.eligible and "Confirm with seller" in r.issues[0]


def test_ineligible_cheap_offer_does_not_set_the_price_scale(company):
    ranked = rank_products(
        [product("P1", 100), product("P2", 1)],
        [assessment("P1", R1="met"), assessment("P2", R1="unmet")],
        REQS, ProcurementRequest(product="laptop"), company,
    )
    assert [r.product.id for r in ranked] == ["P1", "P2"]
    assert ranked[0].price_score == ranked[1].price_score == 100


def test_exchange_rate(company):
    company.exchange_rates = {"usd": 50}
    (r,) = rank_products(
        [product(price=10, currency="USD")], [], REQS, ProcurementRequest(product="laptop"), company
    )
    assert r.unit_price == 500


def test_weights_must_not_all_be_zero():
    with pytest.raises(ConfigError):
        Weights(price=0, specs=0, value=0).normalised()


# --- guardrails ---------------------------------------------------------------
def test_plan_keeps_the_users_requirements_first(state):
    ok, plan = Guardrails(state).plan(
        output({"requirements": [{"id": "X", "text": "Touch screen", "must_have": True}],
                "queries": ["a", "a", " "]})
    )
    assert ok and plan.queries == ["a"]
    assert [(r.id, r.must_have) for r in plan.requirements] == [
        ("R1", True), ("R2", True), ("R3", False), ("R4", False)
    ]


def test_invalid_json_is_sent_back(state):
    ok, feedback = Guardrails(state).plan(output("sorry, no idea"))
    assert not ok and "JSON" in feedback


def test_candidates_must_come_from_the_search_and_several_websites(state):
    for url in ("https://a.example/1", "https://a.example/2", "https://b.example/1"):
        state.found_urls[normalise_url(url)] = url
    guard = Guardrails(state)

    def pick(*urls):
        return output({"candidates": [{"title": "t", "url": u, "reason": "r"} for u in urls]})

    assert guard.candidates(pick("https://invented.example/1"))[0] is False
    ok, feedback = guard.candidates(pick("https://a.example/1", "https://a.example/2"))
    assert not ok and "two different websites" in feedback
    both = pick("https://A.example/1/", "https://b.example/1#x")
    ok, feedback = guard.candidates(both)
    assert not ok and "choose up to 3" in feedback
    ok, result = guard.candidates(both)
    assert ok and [c.url for c in result.candidates] == ["https://a.example/1", "https://b.example/1"]


def test_spare_pages_replace_failed_ones(company):
    request = DEMO_REQUEST.model_copy(update={"max_products": 3})
    services = demo_services()
    real = services.scrapegraph.extract

    def extract(prompt, *, url="", **kw):
        if "store-b" in url:
            return SimpleNamespace(status="error", data=None, error="HTTP 504")
        return real(prompt, url=url, **kw)

    services.scrapegraph = SimpleNamespace(extract=extract)
    state, _ = run_procurement(request, company, services, verbose=False)
    assert len(state.candidates) == 5
    assert len(state.pages) == 3 and len(state.failed_pages) == 1


def test_missing_checks_are_asked_for_once_then_unknown(state):
    state.requirements, state.products = REQS, [product()]
    guard = Guardrails(state)
    answer = output({"assessments": [
        {"product_id": "P1", "pros": [], "cons": [],
         "checks": [{"requirement_id": "R1", "status": "met", "evidence": "16 GB"}]}]})
    assert guard.assessments(answer)[0] is False
    ok, result = guard.assessments(answer)
    assert ok and [c.status for c in result.assessments[0].checks] == ["met", "unknown"]


# --- tools --------------------------------------------------------------------
def test_search_applies_the_company_settings(state):
    seen = {}

    class Client:
        def search(self, query, **options):
            seen.update(options)
            return {"results": [{"title": "t", "url": "javascript:alert(1)"},
                                {"title": "t", "url": "https://a.example/1", "content": None}]}

    rows = json.loads(TavilySearchTool(state=state, client=Client()).run(query="q"))
    assert [r["url"] for r in rows] == ["https://a.example/1"]
    assert seen["country"] == "egypt" and "facebook.com" in seen["exclude_domains"]
    assert "include_domains" not in seen


def test_scraper_refuses_unknown_urls_and_caches(state):
    calls = []

    class Client(DemoScrapeGraph):
        def extract(self, prompt, *, url="", **kw):
            calls.append(url)
            return super().extract(prompt, url=url, **kw)

    tool = ScrapeGraphTool(state=state, client=Client())
    url = "https://store-a.example/orion-workbook-14"
    assert "Refused" in tool.run(url=url)
    state.candidates.append(normalise_url(url))
    assert tool.run(url=url) == tool.run(url=url)
    assert calls == [url]


def test_scraper_falls_back_to_tavily(state):
    url = "https://missing.example/1"
    state.candidates.append(normalise_url(url))
    tavily = DemoTavily()
    tavily.extract = lambda u, **k: {"results": [{"raw_content": "Price 1,234 EGP"}]}
    tool = ScrapeGraphTool(state=state, client=DemoScrapeGraph(), fallback=tavily)
    assert "1,234" in tool.run(url=url)
    assert state.pages[normalise_url(url)].tool == "Tavily Extract"


def test_scraper_rejects_listing_pages(state):
    url = "https://a.example/laptops"
    state.candidates.append(normalise_url(url))
    data = SimpleNamespace(json_data={"is_single_product": False, "title": "Laptops"}, raw=None)
    client = SimpleNamespace(
        extract=lambda *a, **k: SimpleNamespace(status="success", data=data, error=None)
    )
    tool = ScrapeGraphTool(state=state, client=client, fallback=DemoTavily())
    assert "not a page of one product" in tool.run(url=url)
    assert not state.pages


def test_fallback_text_is_extracted_not_truncated(state):
    """Seen on noon.com: the first 1,200 characters were only the site's menu."""
    url = "https://blocked.example/1"
    state.candidates.append(normalise_url(url))
    state.limits = COMPACT
    menu_first = "Menu item\n" * 500 + "Price 1,234 EGP"
    tavily = DemoTavily()
    tavily.extract = lambda u, **k: {"results": [{"raw_content": menu_first}]}

    def extract(prompt, *, markdown=None, url=None, **_):
        if not markdown:
            return SimpleNamespace(status="error", data=None, error="HTTP 504")
        assert "1,234" in markdown
        data = SimpleNamespace(json_data={"title": "T", "price": 1234, "specs": []}, raw=None)
        return SimpleNamespace(status="success", data=data, error=None)

    tool = ScrapeGraphTool(state=state, client=SimpleNamespace(extract=extract), fallback=tavily)
    assert json.loads(tool.run(url=url))["price"] == 1234

    broken = SimpleNamespace(
        extract=lambda *a, **k: SimpleNamespace(status="error", data=None, error="HTTP 504")
    )
    state.pages.clear()
    tool = ScrapeGraphTool(state=state, client=broken, fallback=tavily)
    assert "Scraping failed" in tool.run(url=url)


def test_only_pages_with_a_price_count(state):
    add_page(state, "https://a.example/1", '{"price": null}')
    add_page(state, "https://b.example/1", '{"price": 10}')
    assert state.products_missing == state.request.max_products - 1


def test_websites_take_turns():
    assert spread_over_websites(["https://a/1", "https://a/2", "https://b/1"]) == [
        "https://a/1", "https://b/1", "https://a/2"
    ]


@pytest.mark.parametrize(
    "text, price",
    [("EGP 12,999.50", 12999.5), ("12.999,50 EGP", 12999.5), ("12 999 EGP", 12999),
     ("السعر ١٢٬٩٩٩ جنيه", 12999), ('{"price": 38500.0}', 38500), ("Now 38500.", 38500)],
)
def test_prices_are_found_in_common_formats(text, price):
    assert price in numbers_in(text)


def test_price_is_checked_against_the_extracted_price_field():
    page = ScrapedPage("u", "ScrapeGraph", '{"price": 38500, "specs": [{"value": "1 TB"}]}')
    assert page.shows_price(38500) and not page.shows_price(1)


# --- report and config --------------------------------------------------------
def test_report_escapes_scraped_text_and_handles_no_recommendation(state):
    bad = product(price=None).model_copy(update={"title": "<script>alert(1)</script>"})
    add_page(state, bad.url, "x")
    state.requirements = REQS
    state.ranking = rank_products([bad], [], REQS, state.request, state.company)
    html = render_report(state, ReportContent(
        executive_summary="s", market_overview="m", comparison_notes=[],
        recommended_product_id=None, recommendation="Nothing qualifies", risks=[], next_steps=[]))
    assert "<script>" not in html and "No offer qualifies" in html


def test_missing_keys(monkeypatch):
    for key in ("TAVILY_API_KEY", "SGAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert missing_keys("gemini/x") == [
        "GEMINI_API_KEY or GOOGLE_API_KEY", "TAVILY_API_KEY", "SGAI_API_KEY"]
    monkeypatch.setenv("GOOGLE_API_KEY", "k")
    assert "TAVILY_API_KEY" in missing_keys("gemini/x")[0]
