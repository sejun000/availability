# Makefile for cache_sim and trace_parser

CXX      = g++
CXXFLAGS = -Wall -O2 -std=c++11 -g -rdynamic

# 필요한 Boost 라이브러리 링크 옵션 (사용하는 경우)
LIBS     = -lboost_stacktrace_backtrace -lbacktrace -ldl

TARGET   = cache_sim
SOURCES  = cache_sim.cpp trace_parser.cpp allocator.cpp icache.cpp lru_cache.cpp fifo_cache.cpp log_fifo_cache.cpp
OBJECTS  = $(SOURCES:.cpp=.o)

TARGET2  = trace_replayer
SOURCES2 = trace_replayer.cpp trace_parser.cpp
OBJECTS2 = $(SOURCES2:.cpp=.o)

all: $(TARGET) $(TARGET2)

$(TARGET2): $(OBJECTS2)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(OBJECTS2) $(LIBS)
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) $(OBJECTS2) $(TARGET2)
