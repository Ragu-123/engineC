CXX = g++
CXXFLAGS = -O3 -mavx2 -mfma -fopenmp -march=native -ffast-math -std=c++17 -Iinclude -Wall -Wextra
LDFLAGS = -fopenmp

SRCS = src/main.cpp src/gemma4.cpp src/safetensors.cpp src/tokenizer.cpp
OBJS = $(SRCS:.cpp=.o)
TARGET = gemma4_cpu_engine

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
