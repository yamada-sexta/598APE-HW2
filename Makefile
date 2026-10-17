CXX := g++
OPTFLAGS ?= -O3 -march=native -ffp-contract=off
CXXFLAGS := -std=c++17 -Wall -Wextra -pthread -I include -I src -I. $(OPTFLAGS) -MMD -MP

# Directories
SRC_DIR := src
INC_DIR := include
OBJ_DIR := obj
BENCH_DIR := benchmark

# Find all cpp files in src directory
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
# Generate object file names
OBJS := $(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Benchmark program
BENCH_SRC := $(BENCH_DIR)/genetic_benchmark.cpp
BENCH_BIN := genetic_benchmark

TEST_BINS := correctness training_trace check_fast_math check_rng

# Default target
all: directories $(BENCH_BIN)

# Create necessary directories
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BENCH_DIR)

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | directories
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/fast_math.o: CXXFLAGS += -fno-trapping-math -mprefer-vector-width=512

$(OBJS): Makefile

# build benchmark lib 
$(BENCH_BIN): $(BENCH_SRC) $(OBJS)
	@echo "Building benchmark program..."
	@$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@

correctness training_trace: %: tests/%.cpp $(OBJS)
	@$(CXX) $(CXXFLAGS) $^ -o $@

check_fast_math: tests/check_fast_math.cpp $(OBJ_DIR)/fast_math.o
	@$(CXX) $(CXXFLAGS) $^ -o $@

check_rng: tests/check_rng.cpp $(SRC_DIR)/philox_rng.cpp
	@$(CXX) $(CXXFLAGS) $^ -o $@

-include $(OBJS:.o=.d) $(BENCH_BIN).d $(TEST_BINS:=.d)

test: correctness training_trace
	@./correctness
	@./training_trace > training_trace.txt
	@diff -u tests/fixtures/training_trace.txt training_trace.txt && echo "training trace matches"

check-math: check_fast_math
	@./check_fast_math

check-rng: check_rng
	@./check_rng

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -rf $(OBJ_DIR) $(BENCH_BIN) $(BENCH_BIN).d $(TEST_BINS) $(TEST_BINS:=.d) training_trace.txt

# Clean and rebuild
rebuild: clean all

.PHONY: all clean rebuild directories test check-math check-rng
