from pytest import raises as assert_raises
from graphgallery.utils import raise_if_kwargs, assert_kind


def test_raise_if_kwargs():
    raise_if_kwargs(None)
    assert_raises(TypeError, raise_if_kwargs, name="keywords")

def test_assert_kind():
    assert_kind('T')
    assert_kind('P')
    assert_raises(ValueError, assert_kind, None)

