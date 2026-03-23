from src.eval.metrics import exact_match, f1_token


def test_exact_match_normalizes():
    assert exact_match("Paris.", "paris") == 1


def test_f1_token_perfect_match():
    assert f1_token("retrieval-augmented generation", "retrieval-augmented generation") == 1.0


def test_f1_token_partial_overlap():
    # pred: ["hello", "world"], gold: ["hello", "there"] => overlap=["hello"] => precision=0.5, recall=0.5 => F1=0.5
    assert abs(f1_token("hello world", "hello there") - 0.5) < 1e-9

