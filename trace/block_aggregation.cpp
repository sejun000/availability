#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>

// main 함수
int main(int argc, char* argv[]) {
    if(argc < 3) {
        std::cerr << "Usage: " << argv[0] << " capacity.csv trace.csv" << std::endl;
        return 1;
    }

    std::string capacityFile = argv[1];
    std::string traceFile = argv[2];

    // 1. capacity.csv 파일 읽기
    std::ifstream capFile(capacityFile);
    if(!capFile) {
        std::cerr << "Error opening capacity file: " << capacityFile << std::endl;
        return 1;
    }

    // 각 device의 capacity를 순서대로 저장 (byte 단위)
    std::vector<long long> capacities;
    std::string line;
    while(std::getline(capFile, line)) {
        if(line.empty()) continue;
        std::istringstream iss(line);
        std::string devIdStr, capStr;
        if(!std::getline(iss, devIdStr, ',')) continue;
        if(!std::getline(iss, capStr, ',')) continue;
        try {
            long long cap = std::stoll(capStr);
            capacities.push_back(cap);
        } catch (...) {
            std::cerr << "Error parsing capacity in line: " << line << std::endl;
            continue;
        }
    }
    capFile.close();

    // 2. prefix sum 계산
    // prefix[i]는 device 0부터 device (i-1)까지의 용량 합계를 의미합니다.
    std::vector<long long> prefix;
    prefix.push_back(0);
    for (size_t i = 0; i < capacities.size(); i++) {
        prefix.push_back(prefix.back() + capacities[i]);
    }
    // 예: device_id 0의 누적 용량은 prefix[0] (0),
    //     device_id 1의 누적 용량은 prefix[1] = capacity[0],
    //     device_id 2의 누적 용량은 prefix[2] = capacity[0] + capacity[1], 등

    // 3. trace.csv 파일 읽기 및 재계산된 offset 출력
    std::ifstream inTrace(traceFile);
    if(!inTrace) {
        std::cerr << "Error opening trace file: " << traceFile << std::endl;
        return 1;
    }

    // 타임스탬프 범위: 10일 (10 days in microseconds)
    //const long long TEN_DAYS_US = 864000000000LL;
    // 타임스탬프 범위: 1일 (1 days in microseconds)
    const long long TEN_DAYS_US = 86400000000LL;
    bool firstLine = true;
    long long baseTimestamp = 0; // 첫 행의 timestamp

    // 각 행을 읽어서 device_id에 해당하는 누적 용량 + 기존 offset을 계산하고,
    // timestamp가 첫 행 timestamp 기준 10일 이내인 경우에만 처리
    while(std::getline(inTrace, line)) {
        if(line.empty()) continue;
        std::istringstream iss(line);
        std::string deviceIdStr, opType, offsetStr, sizeStr, timestampStr;
        if(!std::getline(iss, deviceIdStr, ',')) continue;
        if(!std::getline(iss, opType, ',')) continue;
        if(!std::getline(iss, offsetStr, ',')) continue;
        if(!std::getline(iss, sizeStr, ',')) continue;
        if(!std::getline(iss, timestampStr)) continue; // 나머지를 timestamp로 사용
        // this version skip the device id >= 100
        if (atoi(deviceIdStr.c_str()) >= 100) {
            continue;
        }
        // timestamp는 마이크로초 단위 (문자열 -> double 후 long long 변환)
        double tsDouble = std::stod(timestampStr);
        long long currentTimestamp = static_cast<long long>(tsDouble);
        if(firstLine) {
            baseTimestamp = currentTimestamp;
            firstLine = false;
        }
        // 첫 행의 timestamp로부터 10일을 초과하면 중단 (정렬되어 있다고 가정)
        if (currentTimestamp - baseTimestamp > TEN_DAYS_US) {
            break;
        }
        
        int deviceId = std::stoi(deviceIdStr);
        long long offset = std::stoll(offsetStr);

        // deviceId가 capacities 범위 내에 있는지 확인
        if(deviceId < 0 || deviceId >= static_cast<int>(capacities.size())) {
            std::cerr << "Invalid device id: " << deviceId << std::endl;
            continue;
        }

        // 재계산: aggregated_offset = (앞의 모든 device 용량 합계) + (현재 offset)
        long long aggregatedOffset = prefix[deviceId] + offset;

        // 출력: device_id, op_type, aggregated_offset, size, timestamp
        std::cout << deviceIdStr << "," << opType << "," 
                  << aggregatedOffset << "," << sizeStr << "," << timestampStr << "\n";
    }
    inTrace.close();

    return 0;
}
