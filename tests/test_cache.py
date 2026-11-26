from serving import cache

def test_cache_roundtrip_and_version():
    cache.flush()
    assert cache.get_cached(42, 5) is None                 # miss
    cache.set_cached(42, 5, "", [{"movieId": 1}])
    assert cache.get_cached(42, 5) == [{"movieId": 1}]     # hit
    old = cache.MODEL_VERSION
    cache.MODEL_VERSION = "v-test"                         # new model version -> old entries ignored
    try:
        assert cache.get_cached(42, 5) is None
    finally:
        cache.MODEL_VERSION = old
        cache.flush()