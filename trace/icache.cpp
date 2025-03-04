#include "icache.h"
#include "lru_cache.h"
ICache* createCache(std::string cache_type, long capacity, int cache_block_size, bool _cache_trace, const std::string &trace_file) {
    if (cache_type == "LRU") {
        return new LRUCache(capacity, cache_block_size, _cache_trace, trace_file);
    }
    return nullptr;
}