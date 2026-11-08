from receipt_vlm.metrics import field_counts, flatten, normalize_value, score

GT = {
    "menu": [{"nm": "Ice Tea", "price": "12.000"}, {"nm": "Rice", "price": "20.000"}],
    "total": {"total_price": "32.000"},
}


def test_normalize_value():
    assert normalize_value("  Ice   TEA ") == "ice tea"
    assert normalize_value("1 234,50") == "1234.5"
    assert normalize_value(1234.5) == "1234.5"
    assert normalize_value("60.000") == normalize_value(60)
    assert normalize_value("FR12345") == "fr12345"


def test_flatten_drops_list_indices_and_empty_values():
    pairs = flatten({"a": [{"b": "x"}, {"b": "y"}], "c": {"d": 1, "e": None, "f": ""}})
    assert pairs == [("a.b", "x"), ("a.b", "y"), ("c.d", "1")]


def test_field_counts():
    pred = {"menu": [{"nm": "Ice Tea", "price": "12.000"}], "total": {"total_price": "30.000"}}
    counts = field_counts(pred, GT)
    assert counts["menu.nm"] == [1, 1, 2]
    assert counts["menu.price"] == [1, 1, 2]
    assert counts["total.total_price"] == [0, 1, 1]


def test_perfect_prediction():
    result = score([GT, GT], [GT, GT], n_boot=50)
    assert result["f1"] == 1.0
    assert result["exact_match"] == 1.0
    assert result["valid_json"] == 1.0
    assert result["f1_ci"] == [1.0, 1.0]


def test_invalid_json_counts_as_empty_prediction():
    result = score([None, GT], [GT, GT], n_boot=50)
    assert result["valid_json"] == 0.5
    assert result["exact_match"] == 0.5
    # 5 true positives out of 5 predicted and 10 expected.
    assert abs(result["f1"] - 10 / 15) < 1e-9
    assert result["per_field"]["menu.nm"]["support"] == 4


def test_list_order_does_not_matter():
    reordered = {"total": {"total_price": "32.000"}, "menu": [GT["menu"][1], GT["menu"][0]]}
    assert score([reordered], [GT], n_boot=10)["exact_match"] == 1.0


def test_bootstrap_interval_contains_estimate():
    preds = []
    gts = []
    for i in range(40):
        gts.append(GT)
        preds.append(GT if i % 3 else {"total": {"total_price": "0"}})
    result = score(preds, gts, n_boot=200)
    low, high = result["f1_ci"]
    assert low <= result["f1"] <= high
    assert low < high
