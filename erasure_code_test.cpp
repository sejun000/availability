#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern "C" {
    #include <isa-l.h>
}

#define CHUNK_SIZE (4 * 1024) // 256KB
#define N_MIN 2
#define N_MAX 48

int main() {
    int k; // Number of parity chunks
    printf("Enter the number of parity chunks (k): ");
    scanf("%d", &k);

    if (k <= 0) {
        printf("Number of parity chunks must be positive.\n");
        return -1;
    }

    for (int n = N_MIN; n <= N_MAX; n++) {
        int data_chunks = n;
        int parity_chunks = k;
        int total_chunks = data_chunks + parity_chunks;

        uint8_t *data_ptrs[total_chunks];
        uint8_t *encode_matrix;
        uint8_t *g_tbls;

        // Allocate memory for data and parity chunks
        for (int i = 0; i < total_chunks; i++) {
            data_ptrs[i] = (uint8_t *) malloc(CHUNK_SIZE);
            if (data_ptrs[i] == NULL) {
                printf("Memory allocation failed.\n");
                return -1;
            }
            // Initialize data chunks with random data, parity chunks with zeros
            if (i < data_chunks) {
                for (int j = 0; j < CHUNK_SIZE; j++) {
                    data_ptrs[i][j] = rand() % 256;
                }
            } else {
                memset(data_ptrs[i], 0, CHUNK_SIZE);
            }
        }

        // Allocate memory for encode matrix and g_tbls
        encode_matrix = (uint8_t *) malloc(total_chunks * data_chunks);
        g_tbls = (uint8_t *) malloc(32 * data_chunks * parity_chunks);

        if (encode_matrix == NULL || g_tbls == NULL) {
            printf("Memory allocation failed.\n");
            return -1;
        }

        // Generate encode matrix
        gf_gen_rs_matrix(encode_matrix, total_chunks, data_chunks);

        // Generate g_tbls from encode matrix
        ec_init_tables(data_chunks, parity_chunks, &encode_matrix[data_chunks * data_chunks], g_tbls);

        // Measure encoding time
        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        // Perform encoding
        ec_encode_data(CHUNK_SIZE, data_chunks, parity_chunks, g_tbls, data_ptrs, &data_ptrs[data_chunks]);

        clock_gettime(CLOCK_MONOTONIC, &end);

        // Calculate elapsed time in microseconds
        uint64_t elapsed_time = (end.tv_sec - start.tv_sec) * 1000000 +
                                (end.tv_nsec - start.tv_nsec) / 1000;

        printf("n = %d, k = %d, Encoding Time: %lu microseconds\n", n, k, elapsed_time);

        // Free allocated memory
        for (int i = 0; i < total_chunks; i++) {
            free(data_ptrs[i]);
        }
        free(encode_matrix);
        free(g_tbls);
    }

    return 0;
}
