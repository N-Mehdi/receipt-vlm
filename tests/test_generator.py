import random
from collections import Counter
from decimal import Decimal

import pytest
from faker import Faker

from receipt_vlm.data import normalize_cord
from receipt_vlm.invoices import (
    SEEN_TEMPLATES,
    UNSEEN_TEMPLATES,
    eur,
    luhn_complete,
    luhn_valid,
    make_env,
    make_invoice,
    make_siret,
    render_html,
    split_plan,
    vat_number,
)


def invoices(n, seed=0):
    rng = random.Random(seed)
    fake = Faker("fr_FR")
    fake.seed_instance(seed)
    out = []
    for _ in range(n):
        out.append(make_invoice(fake, rng))
    return out


def test_luhn():
    assert luhn_valid("732829320")
    assert not luhn_valid("732829321")
    assert luhn_valid(luhn_complete("12345678"))


def test_siret_and_vat_number_are_valid():
    rng = random.Random(1)
    for _ in range(200):
        siret = make_siret(rng)
        assert len(siret) == 14 and siret.isdigit()
        assert luhn_valid(siret)
        assert luhn_valid(siret[:9])
    # Known pair: SIREN 732829320 has VAT key 44.
    assert vat_number("732829320") == "FR44732829320"


def test_amounts_are_consistent():
    for target, _ in invoices(200):
        total = Decimal("0")
        for item in target["items"]:
            line = Decimal(str(item["unit_price"])) * item["quantity"]
            assert line == Decimal(str(item["amount"]))
            total += Decimal(str(item["amount"]))
        ht = Decimal(str(target["total_ht"]))
        vat = Decimal(str(target["vat_amount"]))
        assert total == ht
        assert ht + vat == Decimal(str(target["total_ttc"]))
        expected_vat = ht * Decimal(str(target["vat_rate"])) / 100
        assert abs(vat - expected_vat) <= Decimal("0.005")
        assert len(target["date"]) == 10
        assert "\n" not in target["seller"]["address"]


def test_split_plan_keeps_unseen_templates_for_test():
    plan = split_plan(1000, 100, 50, 50)
    splits = Counter()
    unseen = 0
    for split, template in plan:
        splits[split] += 1
        if template in UNSEEN_TEMPLATES:
            assert split == "test"
            unseen += 1
    assert splits == {"train": 1000, "validation": 100, "test": 100}
    assert unseen == 50


@pytest.mark.parametrize("template", SEEN_TEMPLATES + UNSEEN_TEMPLATES)
def test_templates_show_annotated_values(template):
    target, display = invoices(1, seed=3)[0]
    html = render_html(make_env(), template, target, display)
    for value in [target["seller"]["siret"], target["invoice_number"], target["client"]["name"]]:
        assert str(value).replace("&", "&amp;").replace("'", "&#39;") in html
    assert eur(display["total_ttc"]) in html


def test_eur_format():
    assert eur(Decimal("1234.5")) == "1 234,50 €"


def test_normalize_cord_makes_item_lists():
    raw = {"menu": {"nm": "A", "sub": {"nm": "B"}}, "total": {"total_price": "1"}}
    assert normalize_cord(raw) == {
        "menu": [{"nm": "A", "sub": [{"nm": "B"}]}],
        "total": {"total_price": "1"},
    }
