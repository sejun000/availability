#include "icache.h"
#include "lru_cache.h"
#include "fifo_cache.h"
#include "log_fifo_cache.h"
ICache* createCache(std::string cache_type, long capacity, int cache_block_size, bool _cache_trace, const std::string &trace_file) {
    if (cache_type == "LRU") {
        return new LRUCache(capacity, cache_block_size, _cache_trace, trace_file);
    }
    else if (cache_type == "FIFO") {
        return new FIFOCache(capacity, cache_block_size, _cache_trace, trace_file);
    }
    else if (cache_type == "LOG_FIFO") {
        return new LogFIFOCache(capacity, cache_block_size, _cache_trace, trace_file);
    }
    else {
        assert(false);
    }
    return nullptr;
}

std::tuple<long long, long long, long long> ICache::get_status() {
    
    return std::tuple<long long, long long, long long>(write_size_to_cache, evicted_blocks, write_hit_size);
}