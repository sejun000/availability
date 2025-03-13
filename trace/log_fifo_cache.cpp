#include "log_fifo_cache.h"
#include <cassert>
#include <algorithm>

LogFIFOCache::LogFIFOCache(long capacity, int _cache_block_size, bool _cache_trace, const std::string &trace_file)
    : capacity_(capacity), cache_block_size(_cache_block_size), cache_trace(_cache_trace),
      write_ptr(0)
{
    if (cache_trace) {
        cache_trace_fp = fopen(trace_file.c_str(), "w");
    } else {
        cache_trace_fp = nullptr;
    }
    // 전체 캐시 용량(바이트) = capacity * cache_block_size.
    // 이를 4K 단위로 나누어 원형 버퍼의 크기를 결정합니다.
    size_t num_units = capacity;
    if (num_units == 0) num_units = 1;
    log_buffer.resize(num_units, {0, false});
    // 한 캐시 블록은 cache_block_size/4096개의 4K 엔트리로 구성됨.
}

LogFIFOCache::~LogFIFOCache() {
    if (cache_trace_fp) {
        fclose(cache_trace_fp);
    }
}

bool LogFIFOCache::exists(long key) {
    return mapping.find(key) != mapping.end();
}

void LogFIFOCache::touch(long key, OP_TYPE op_type) {
    // LogFIFOCache는 항상 새로운 데이터를 append 하므로, touch는 별도 동작 없이 무시합니다.
}

void LogFIFOCache::evict_one_block(){
    long old_key = log_buffer[write_ptr].key;
    if (mapping.find(old_key) != mapping.end() && mapping[old_key] == write_ptr) {
        mapping.erase(old_key);
    }
    log_buffer[write_ptr].valid = false;
}

void LogFIFOCache::batch_insert(const std::unordered_set<long> &newBlocks, OP_TYPE op_type) {
    for (auto key : newBlocks) {
        if (exists(key)) {
            auto it = mapping.find(key);
            if (it != mapping.end()) {
                size_t pos = it->second;
                log_buffer[pos].valid = false;
                mapping.erase(it);
            }
        }
        // 새 항목 삽입 전에, write_ptr 위치가 유효하면 그룹 전체 evict
        if (log_buffer[write_ptr].valid) {
            if (write_ptr < log_buffer.size() && log_buffer[write_ptr].valid) {
                evict_one_block();
                evicted_blocks++;
            }
        }
        log_buffer[write_ptr] = { key, true };
        mapping[key] = write_ptr;
        write_ptr = (write_ptr + 1) % log_buffer.size();
        write_size_to_cache += cache_block_size;
    }
}

bool LogFIFOCache::is_cache_filled() {
    // 로그 버퍼의 모든 슬롯이 유효하면 캐시가 가득 찼다고 봅니다.
    for (const auto &entry : log_buffer) {
        if (!entry.valid)
            return false;
    }
    return true;
}

int LogFIFOCache::get_block_size() {
    return cache_block_size;
}

void LogFIFOCache::print_cache_trace(long long lba_offset, int lba_size, OP_TYPE op_type) {
    if (cache_trace_fp) {
        long start_block = static_cast<long>(lba_offset / cache_block_size);
        long end_block = static_cast<long>((lba_offset + lba_size) / cache_block_size);
        // print trace as csv format
        for (long block = start_block; block <= end_block; block++) {
            auto iter = mapping.find(block);
            const int DUMMY_VALUE = 0;
            std::string op_string = "W";
            if (op_type == OP_TYPE::READ) {
                op_string = "R";
            }
            if (iter != mapping.end()) {
                size_t id = iter->second;
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

size_t LogFIFOCache::size() {
    return mapping.size();
}