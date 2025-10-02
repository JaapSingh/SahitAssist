from sahitassist.normalization import normalize_line

def test_normalization_simple():
    mapping = {"A": "ਅ", "B": "ਬ"}
    assert normalize_line("AB", mapping) == "ਅਬ"