#include "cache_sim.h"
#include "trace_parser.h"
#include "icache.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <tuple>   // std::tuple

// =======================
// 캐시 hit 및 정책 관련 함수 구현
// =======================


// 주어진 lba 범위를 블록 단위로 나누어 캐시 내에 존재하는 블록의 겹치는 바이트 수를 계산
std::tuple<long long, int, int, int, int> check_cache_hit(ICache &cache, long long lba_offset, int lba_size, int block_size, OP_TYPE op_type) {
    long start_block = static_cast<long>(lba_offset / block_size);
    long end_block = static_cast<long>((lba_offset + lba_size) / block_size);
    long long hit_bytes = 0;
    std::unordered_set<long> request_blocks;
    for (long block = start_block; block <= end_block; block++) {
        request_blocks.insert(block);
    }
    int touched_cache_blocks = 0;
    int write_size_to_cache = 0;
    int miss_blocks = 0;
    int read_miss_blocks = 0;
    for (long block : request_blocks) {
        long long block_start = static_cast<long long>(block) * block_size;
        long long block_end = block_start + block_size;
        long long req_start = lba_offset;
        long long req_end = lba_offset + lba_size;
        long long left_offset = std::max(block_start, req_start);
        long long right_offset = std::min(block_end, req_end);
        if (cache.exists(block)) {
            int hit_bytes_per_block = 0;
            if (right_offset > left_offset)
                hit_bytes_per_block = right_offset - left_offset;
            hit_bytes += hit_bytes_per_block;
            write_size_to_cache += hit_bytes_per_block;
            cache.touch(block, op_type);
            touched_cache_blocks++;
        }
        else {
            write_size_to_cache += block_size;
            if (right_offset != block_end || left_offset != block_start) {
                read_miss_blocks++;
            }
        }
    }
    miss_blocks = request_blocks.size() - touched_cache_blocks;
    return std::tuple<long long, int, int, int, int>(hit_bytes, touched_cache_blocks, write_size_to_cache, miss_blocks, read_miss_blocks);
}

// LRU 정책: 주어진 lba 범위의 블록들을 캐시에 추가
int lru_cache_policy(ICache& cache, long long lba_offset, int lba_size, int hit_blocks, OP_TYPE op_type) {
    int block_size = cache.get_block_size();
    long start_block = static_cast<long>(lba_offset / block_size);
    long end_block = static_cast<long>((lba_offset + lba_size) / block_size);
    std::unordered_set<long> newBlocks;
    for (long block = start_block; block <= end_block; block++) {
        newBlocks.insert(block);
    }
    return cache.batch_insert(newBlocks, hit_blocks, op_type);
}

// Read/Write hit ratio 계산 (퍼센트)
void calc_hit_ratio(long long read_hit_size, long long total_read_size,
                    long long write_hit_size, long long total_write_size,
                    double &read_hit_ratio, double &write_hit_ratio) {
    read_hit_ratio = (total_read_size > 0) ? (static_cast<double>(read_hit_size) / total_read_size) * 100 : 0;
    write_hit_ratio = (total_write_size > 0) ? (static_cast<double>(write_hit_size) / total_write_size) * 100 : 0;
}

// =======================
// main() 함수
// =======================

void print_stats(bool intermeidate, long long total_read, long long total_write, long long total_read_size, long long total_write_size, long long read_hit_size, long long write_hit_size, long long cache_write_size, long long cold_tier_write_size, long long cold_tier_read_size, long max_cache_blocks, size_t cache_size) {
    double final_read_hit_ratio, final_write_hit_ratio;
    calc_hit_ratio(read_hit_size, total_read_size, write_hit_size, total_write_size, final_read_hit_ratio, final_write_hit_ratio);
    
    if (intermeidate) {
        std::cout << "\nIntermediate Stats" << std::endl;
    }
    else {
        std::cout << "\nFinal Stats" << std::endl;
    }
    std::cout << "\nCurrent Cache Hit Ratios:" << std::endl;
    std::cout << "Read Cache Hit Ratio: " << final_read_hit_ratio << "%" << std::endl;
    std::cout << "Write Cache Hit Ratio: " << final_write_hit_ratio << "%" << std::endl;
    std::cout << "total_read = " << total_read << ", total_write = " << total_write << std::endl;
    std::cout << "total_read_bytes = " << total_read_size << ", total_write_bytes = " << total_write_size << std::endl;
    std::cout << "cache size = " << max_cache_blocks << std::endl;
    std::cout << "current cache size = " << cache_size << std::endl;
    std::cout << "cache_write_size = " << cache_write_size << std::endl;
    std::cout << "cold_tier_write_size = " << cold_tier_write_size << std::endl;
    std::cout << "cold_tier_read_size = " << cold_tier_read_size << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " trace_file cache_size [--block_size N] [--policy all|write-only] [--trace_format csv|blktrace] [--cache_trace]" << std::endl;
        return 1;
    }
    std::string trace_file = argv[1];
    long cache_size = std::stol(argv[2]);
    int block_size = 65536; // 기본 블록 크기
    std::string policy = "all";
    std::string trace_format = "csv";
    std::string cache_trace_output = "";
    bool cache_trace = false;

    // 추가 인자 파싱
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--block_size" && i + 1 < argc) {
            block_size = std::stoi(argv[++i]);
        } else if (arg == "--policy" && i + 1 < argc) {
            policy = argv[++i];
        } else if (arg == "--trace_format" && i + 1 < argc) {
            trace_format = argv[++i];
        } else if (arg == "--cache_trace" && i + 1 < argc) {
            cache_trace_output = argv[++i];
            cache_trace = true;
        }
    }
    // Factory 함수를 이용해 적절한 TraceParser 생성
    ITraceParser* parser = createTraceParser(trace_format);
    long max_cache_blocks = cache_size / block_size;
    printf("max_cache_blocks = %ld\n", max_cache_blocks);
    ICache* cache = createCache("LRU", max_cache_blocks, block_size, cache_trace, cache_trace_output);
    // 통계 변수 초기화
    long long total_read = 0, total_write = 0;
    long long total_read_size = 0, total_write_size = 0;
    long long read_hit_size = 0, write_hit_size = 0;
    long long cache_write_size = 0, cold_tier_write_size = 0, cold_tier_read_size = 0;
    
    std::ifstream infile(trace_file);
    if (!infile) {
        std::cerr << "File Error" << std::endl;
        std::cerr << "Cannot open file: " << trace_file << std::endl;
        return 1;
    }
    
    std::string line;
    long long line_count = 0;
    const long long line_count_limit = 4000000000;
    
    while (std::getline(infile, line) && line_count < line_count_limit) {
        line_count++;
        if (line_count % 1000000 == 0) {
            print_stats(true, total_read, total_write, total_read_size, total_write_size, read_hit_size, write_hit_size, cache_write_size, cold_tier_write_size, cold_tier_read_size, max_cache_blocks, cache->size());
        }
        
        // 사용자 구현 parse_trace 함수 호출
        ParsedRow parsed = parser->parseTrace(line);
        // printf ("parsed.dev_id = %s, parsed.op_type = %s, parsed.lba_offset = %lld, parsed.lba_size = %d, parsed.timestamp = %f\n", parsed.dev_id.c_str(), parsed.op_type.c_str(), parsed.lba_offset, parsed.lba_size, parsed.timestamp);
        if (parsed.dev_id.empty()) {
            continue;
        }
        long long hit_size;
        int hit_blocks;
        int write_bytes_to_cache;
        int miss_blocks;
        int read_miss_blocks;
        
        if (parsed.op_type == "R" || parsed.op_type == "RS") {
            std::tie(hit_size, hit_blocks, write_bytes_to_cache, miss_blocks, read_miss_blocks)= check_cache_hit(*cache, parsed.lba_offset, parsed.lba_size, block_size, OP_TYPE::READ);
            //if (cache.is_cache_filled()) {
                total_read++;
                total_read_size += parsed.lba_size;
                read_hit_size += hit_size;
            //}
            if (policy == "all" || policy == "read-only") {
                lru_cache_policy(*cache, parsed.lba_offset, parsed.lba_size, hit_blocks, OP_TYPE::READ) * block_size;
            }
        } else if (parsed.op_type == "W" || parsed.op_type == "WS") {
            std::tie(hit_size, hit_blocks, write_bytes_to_cache, miss_blocks, read_miss_blocks)= check_cache_hit(*cache, parsed.lba_offset, parsed.lba_size, block_size, OP_TYPE::WRITE);
            //if (cache.is_cache_filled()) {
                total_write++;
                total_write_size += parsed.lba_size;
                write_hit_size += hit_size;
                cache_write_size += write_bytes_to_cache;
                cold_tier_write_size += block_size * miss_blocks;
                cold_tier_read_size += block_size * read_miss_blocks;
            //}
            if (policy == "all" || policy == "write-only") {
                lru_cache_policy(*cache, parsed.lba_offset, parsed.lba_size, hit_blocks, OP_TYPE::WRITE) * block_size;
            }
            if (policy == "write-only") {
                cache->print_cache_trace(parsed.lba_offset, parsed.lba_size, OP_TYPE::WRITE);
            }

        }
    }
    
    double final_read_hit_ratio, final_write_hit_ratio;
    calc_hit_ratio(read_hit_size, total_read_size, write_hit_size, total_write_size, final_read_hit_ratio, final_write_hit_ratio);
    
    print_stats(false, total_read, total_write, total_read_size, total_write_size, read_hit_size, write_hit_size, cache_write_size, cold_tier_write_size, cold_tier_read_size, max_cache_blocks, cache->size());
    
    return 0;
}
