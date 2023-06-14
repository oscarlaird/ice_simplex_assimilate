from hypothesis import given
from hypothesis import strategies as st

@given(st.integers(), st.integers())
def test_addition_is_commutative(x, y):
    assert x + y == y + x

