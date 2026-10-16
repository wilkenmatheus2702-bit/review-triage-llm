import pandas as pd

from review_triage import data


def test_clean_text_normalises_whitespace():
    assert data.clean_text("  Ótimo\r\nproduto \t chegou  ") == "Ótimo produto chegou"
    assert data.clean_text(None) == ""
    assert data.clean_text(float("nan")) == ""


def test_text_hash_ignores_case_and_spacing():
    assert data.text_hash("Ótimo  produto") == data.text_hash("ótimo produto")
    assert data.text_hash("a") != data.text_hash("b")


def _raw(tmp_path):
    pd.DataFrame(
        {
            "review_id": ["r1", "r1", "r2", "r3", "r4"],
            "order_id": ["o1", "o1", "o2", "o3", "o4"],
            "review_score": [1, 1, 5, 4, 3],
            "review_comment_title": [None, None, "Top", None, None],
            "review_comment_message": [
                "primeira versao",
                "nao\r\nrecebi",
                "chegou rapido",
                None,
                "veio atrasado",
            ],
            "review_creation_date": ["2018-01-01"] * 5,
            "review_answer_timestamp": [
                "2018-01-01 10:00:00",
                "2018-01-02 10:00:00",
                "2018-01-01 10:00:00",
                "2018-01-01 10:00:00",
                "2018-01-01 10:00:00",
            ],
        }
    ).to_csv(tmp_path / "olist_order_reviews_dataset.csv", index=False)
    pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o4"],
            "order_status": ["shipped", "delivered", "delivered"],
            "order_delivered_customer_date": [None, "2018-01-10 10:00:00", "2018-01-20 10:00:00"],
            "order_estimated_delivery_date": [
                "2018-01-15 00:00:00",
                "2018-01-15 00:00:00",
                "2018-01-15 00:00:00",
            ],
        }
    ).to_csv(tmp_path / "olist_orders_dataset.csv", index=False)
    return tmp_path


def test_load_reviews(tmp_path):
    df = data.load_reviews(_raw(tmp_path)).set_index("review_id")
    assert set(df.index) == {"r1", "r2", "r4"}  # r3 has no comment
    assert df.loc["r1", "text"] == "nao recebi"  # latest answer kept, whitespace cleaned
    assert df.loc["r2", "text"] == "Top. chegou rapido"  # title prepended
    assert pd.isna(df.loc["r1", "delivered_late"])  # not delivered
    assert not df.loc["r2", "delivered_late"] and df.loc["r4", "delivered_late"]


def test_stratified_sample_is_reproducible(tmp_path):
    df = pd.DataFrame({"review_id": range(100), "stars": [1] * 50 + [5] * 50})
    a = data.stratified_sample(df, {1: 10, 5: 80})
    b = data.stratified_sample(df, {1: 10, 5: 80})
    assert a.equals(b)
    assert a["stars"].value_counts().to_dict() == {1: 10, 5: 50}  # capped at what exists


def test_main(tmp_path):
    raw = _raw(tmp_path)
    assert data.main(["--raw", str(raw), "--out", str(tmp_path / "out.csv")]) == 0
    assert len(pd.read_csv(tmp_path / "out.csv")) == 3


def test_committed_sample_matches_documentation():
    df = pd.read_csv("data/reviews.csv")
    assert len(df) == 2000 and df["review_id"].is_unique
    gold = pd.read_csv("data/gold.csv")
    assert len(gold) == 200 and gold["review_id"].isin(df["review_id"]).all()
