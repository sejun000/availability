# Makefile for cache_sim and trace_parser

CXX      = g++
CXXFLAGS = -Wall -O2 -std=c++11

TARGET   = cache_sim
SOURCES  = cache_sim.cpp trace_parser.cpp allocator.cpp icache.cpp lru_cache.cpp
OBJECTS  = $(SOURCES:.cpp=.o)

TARGET2 = trace_replayer
SOURCES2 = trace_replayer.cpp trace_parser.cpp
OBJECTS2  = $(SOURCES2:.cpp=.o)

all: $(TARGET) $(TARGET2)

$(TARGET2): $(OBJECTS2)
	$(CXX) $(CXXFLAGS) -o $(TARGET2) $(OBJECTS2)
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET) $(OBJECTS2) $(TARGET2)

