#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "trace_parser.h"

#define BLOCK_SIZE 4096
#define SECTOR_SIZE 512

// IO operation을 나타내는 구조체 (예: lba, offset, size, op 타입 등)
typedef struct {
    unsigned long lba;   // 논리 블록 주소
    unsigned long offset; // 해당 블록 내의 오프셋 (바이트 단위)
    size_t size;          // write할 데이터 크기 (바이트 단위)
    char op;              // 'w' for write, 'r' for read 등 (여기서는 write만 사용)
} io_operation;

// 이미 구현되어 있는 trace 읽기 함수 (여기서는 빈 함수로 둡니다)
void read_trace(const char *trace_file, io_operation **ops, size_t *num_ops) {
    // trace 파일에서 IO operation들을 파싱하는 부분 (이미 구현되어 있다고 가정)
    *ops = NULL;
    *num_ops = 0;
}

long long align(long long value, long long alignment) {
    return ((value + alignment - 1) / alignment) * alignment;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <trace file> <target disk file> <trace format(blktrace/csv)>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *trace_file = argv[1];
    const char *disk_file = argv[2];
    const char *trace_format = argv[3];

    // Factory 함수를 이용해 적절한 TraceParser 생성
    ITraceParser* parser = createTraceParser(trace_format);
    std::ifstream infile(trace_file);
    if (!infile) {
        std::cerr << "Cannot open file: " << trace_file << std::endl;
        return 1;
    }
    std::string line;
    // 디스크 파일을 O_DIRECT 옵션으로 open (버퍼 없이 write)
    int fd = open(disk_file, O_WRONLY | O_DIRECT);
    if (fd < 0) {
        perror("open");
        return EXIT_FAILURE;
    }
    long long line_count = 0;
    long long total_write_bytes = 0;
    const long long line_count_limit = 2750000000;
    
    void *buf;
    if (posix_memalign(&buf, BLOCK_SIZE, 1024ULL * 1024ULL * 256) != 0) {
        fprintf(stderr, "posix_memalign failed\n");
        close(fd);
        return EXIT_FAILURE;
    }
    while (std::getline(infile, line) && line_count < line_count_limit) {
        ParsedRow parsed = parser->parseTrace(line);
        // printf ("parsed.dev_id = %s, parsed.op_type = %s, parsed.lba_offset = %lld, parsed.lba_size = %d, parsed.timestamp = %f\n", parsed.dev_id.c_str(), parsed.op_type.c_str(), parsed.lba_offset, parsed.lba_size, parsed.timestamp);
        line_count++;
        if (line_count % 1000000 == 0) {
            printf("line_count: %lld\n", line_count);
            printf("total_write_bytes as GB: %lf\n", (double)total_write_bytes / 1000.0 / 1000.0 / 1000.0);
        }
        if (parsed.dev_id.empty()) {
            continue;
        }
        // 각 IO operation에 대해 write 수행 (write인 경우만)
        if (parsed.op_type == "W" || parsed.op_type == "WS") {
            // O_DIRECT를 사용할 때는 메모리 alignment가 필요함 (일반적으로 BLOCK_SIZE 단위)
            off_t pos = parsed.lba_offset;
            // target 위치 계산 (예: lba*BLOCK_SIZE + offset)
            off_t right_pos = pos + parsed.lba_size;
            off_t size = 0;
            pos = align(pos, BLOCK_SIZE);
            right_pos = align(right_pos, BLOCK_SIZE);
            size = right_pos - pos;
            if (lseek(fd, pos, SEEK_SET) < 0) {
                printf("lseek failed, lba_offset: %ld, lba_size: %ld pos: %ld\n", parsed.lba_offset, parsed.lba_size, pos);
                perror("lseek");
                free(buf);
                close(fd);
                return EXIT_FAILURE;
            }

            // write 수행
            ssize_t written = write(fd, buf, size);
            total_write_bytes += written;
            if (written < 0) {
                perror("write");
                free(buf);
                close(fd);
                return EXIT_FAILURE;
            }
        }
    }
    free(buf);
    close(fd);
    return EXIT_SUCCESS;
}
