#pragma once
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cassert>
#include "common.h"

struct CacheEntry {
    std::list<long>::iterator iter;
    size_t allocated_id;
};

class ICache {
public:
    // 생성자: capacity는 블록 단위 최대 개수
    ~ICache(){}
    
    virtual bool exists(long key) = 0;
    virtual void insert(long key, OP_TYPE op_type) = 0;
    virtual void touch(long key, OP_TYPE op_type) = 0;
    virtual int batch_insert(const std::unordered_set<long> &newBlocks, int hit_blocks, OP_TYPE op_type) = 0;
    virtual bool is_cache_filled() = 0;
    virtual int get_block_size() = 0;
    virtual void print_cache_trace(long long lba_offset, int lba_size, OP_TYPE op_type) = 0;
    virtual size_t size() = 0;
};

ICache* createCache(std::string cache_type, long capacity, int cache_block_size, bool _cache_trace, const std::string &trace_file);