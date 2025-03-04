#include "lru_cache.h"
#include <cassert>

LRUCache::LRUCache(long capacity, int _cache_block_size, bool _cache_trace, const std::string &trace_file) : capacity_(capacity), cache_block_size(_cache_block_size), cache_trace(_cache_trace), allocator(capacity * _cache_block_size) {
    cache_trace_fp = nullptr;
    //printf("LRU cache created with capacity: %ld\n", capacity);
    if (cache_trace) {
        cache_trace_fp = fopen(trace_file.c_str(), "w");
    }
}

LRUCache::~LRUCache() {
    if (cache_trace_fp) {
        fclose(cache_trace_fp);
    }
}

bool LRUCache::exists(long key) {
    return cacheMap.find(key) != cacheMap.end();
}

int LRUCache::get_block_size() {
    return cache_block_size;
}

void LRUCache::print_cache_trace(long long lba_offset, int lba_size, OP_TYPE op_type) {
    if (cache_trace_fp) {
        long start_block = static_cast<long>(lba_offset / cache_block_size);
        long end_block = static_cast<long>((lba_offset + lba_size) / cache_block_size);
        // print trace as csv format
        for (long block = start_block; block <= end_block; block++) {
            CacheEntry entry = cacheMap[block];
            const int DUMMY_VALUE = 0;
            std::string op_string = "W";
            if (op_type == OP_TYPE::READ) {
                op_string = "R";
            }
            if (entry.iter != cacheList.end()) {
                size_t id = entry.allocated_id;
                long long block_start = static_cast<long long>(block) * cache_block_size;
                long long block_end = block_start + cache_block_size;
                long long req_start = lba_offset;
                long long req_end = lba_offset + lba_size;
                long long left_offset = std::max(block_start, req_start);
                long long right_offset = std::min(block_end, req_end);
                size_t cache_lba_offset = id * cache_block_size + left_offset % cache_block_size;
                size_t cache_lba_size = right_offset - left_offset;
                if (cache_lba_size == 0) {
                    continue;
                }
                fprintf(cache_trace_fp, "%ld,%s,%ld,%ld,%ld\n", DUMMY_VALUE, op_string.c_str(), cache_lba_offset, cache_lba_size, DUMMY_VALUE);
            } else {
                assert(false);
            }
        }
    }
}

void LRUCache::touch(long key, OP_TYPE op_type) {
    auto it = cacheMap.find(key);
    if (it != cacheMap.end()) {
        size_t id = it->second.allocated_id;
        cacheList.erase(it->second.iter);
        cacheList.push_back(key);
        CacheEntry cacheEntry = {
            .iter = std::prev(cacheList.end()),
            .allocated_id = id
        };
        cacheMap[key] = cacheEntry;
    }
}

void LRUCache::insert(long key, OP_TYPE op_type) {
    if (exists(key)) {
        touch(key, op_type);
        return;
    }
    while (cacheMap.size() + 1 > static_cast<size_t>(capacity_)) {
        long oldest = cacheList.front();
        cacheList.pop_front();
        auto second = cacheMap[oldest];
        size_t id = second.allocated_id;
        cacheMap.erase(oldest);
        allocator.free(id);
        cache_filled = true;
    }
    cacheList.push_back(key);
    size_t alloc_id = allocator.alloc();
    CacheEntry cacheEntry = {
        .iter = std::prev(cacheList.end()),
        .allocated_id = alloc_id
    };
    cacheMap[key] = cacheEntry;
}

int LRUCache::batch_insert(const std::unordered_set<long> &newBlocks, int hit_blocks, OP_TYPE op_type) {
    int evicted_blocks = 0;
    while (cacheMap.size() + newBlocks.size() - hit_blocks > static_cast<size_t>(capacity_)) {
        long oldest = cacheList.front();
        cacheList.pop_front();
        auto second = cacheMap[oldest];
        size_t id = second.allocated_id;
        cacheMap.erase(oldest);
        allocator.free(id);
        cache_filled = true;
        evicted_blocks++;
    }
    for (long block : newBlocks) {
        if (exists(block)) {
            touch(block, op_type);
        } else {
            cacheList.push_back(block);
            size_t alloc_id = allocator.alloc();
            CacheEntry cacheEntry = {
                .iter = std::prev(cacheList.end()),
                .allocated_id = alloc_id
            };
            cacheMap[block] = cacheEntry;
        }
    }
    return evicted_blocks;
}

bool LRUCache::is_cache_filled() {
    return cache_filled;
}

size_t LRUCache::size(){
    //printf("cacheMap.size() = %ld\n", cacheMap.size());
    return cacheMap.size();
}