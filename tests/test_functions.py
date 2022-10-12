"""
Unit-test functions.py script
"""
import python.functions as f


def test_clean_whitespaces():
    assert f.clean_whitespaces(' hello Mr  President ') == 'hello Mr President'
