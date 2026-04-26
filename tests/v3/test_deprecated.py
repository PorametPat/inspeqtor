import warnings


def old_func():
    warnings.warn("old_func is deprecated", DeprecationWarning, stacklevel=2)


def test_deprecated_function():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        old_func()
        assert len(w) == 1
        assert issubclass(w[-1].category, DeprecationWarning)
        assert "old_func is deprecated" in str(w[-1].message)
