import os

def test_wiki_directories_exist():
    expected_dirs = [
        "wiki/research",
        "wiki/general",
        "wiki/meta"
    ]
    for d in expected_dirs:
        assert os.path.exists(d), f"Directory {d} does not exist"
