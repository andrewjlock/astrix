

def test_backend_import():
    import astrix as at
    at._backend_utils.resolve_backend("np")
    if at._backend_utils.HAS_JAX:
        at._backend_utils.resolve_backend("jnp")




