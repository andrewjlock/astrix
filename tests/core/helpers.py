
def to_text(object):
    """ test object __repr__ and __str__ methods """
    assert isinstance(repr(object), str)
    assert isinstance(str(object), str)
    if hasattr(object, "_xp"):
        assert object.backend == object._xp.__name__


