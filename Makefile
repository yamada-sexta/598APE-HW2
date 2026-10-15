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

# Default target
all: directories $(BENCH_BIN)

# Create necessary directories
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BENCH_DIR)

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/fast_math.o: CXXFLAGS += -fno-trapping-math -mprefer-vector-width=512

$(OBJS): Makefile

# build benchmark lib 
$(BENCH_BIN): $(BENCH_SRC) $(OBJS)
	@echo "Building benchmark program..."
	@$(CXX) $(CXXFLAGS) $< $(OBJS) -o $@

-include $(OBJS:.o=.d) $(BENCH_BIN).d

check-math: tests/check_fast_math.cpp $(OBJ_DIR)/fast_math.o
	@$(CXX) $(CXXFLAGS) $^ -o check_fast_math && ./check_fast_math

test: tests/correctness.cpp tests/training_trace.cpp $(OBJS)
	@$(CXX) $(CXXFLAGS) tests/correctness.cpp $(OBJS) -o correctness && ./correctness
	@$(CXX) $(CXXFLAGS) tests/training_trace.cpp $(OBJS) -o training_trace
	@./training_trace > training_trace.txt
	@diff -u tests/fixtures/training_trace.txt training_trace.txt && echo "training trace matches"

check-rng: tests/check_rng.cpp $(SRC_DIR)/philox_rng.cpp include/philox_engine.h include/custom_distributions.h
	@$(CXX) $(CXXFLAGS) $< $(SRC_DIR)/philox_rng.cpp -o check_rng && ./check_rng

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -rf $(OBJ_DIR) $(BENCH_BIN) $(BENCH_BIN).d check_fast_math check_fast_math.d check_rng check_rng.d correctness correctness.d training_trace training_trace.d training_trace.txt

# Clean and rebuild
rebuild: clean all

.PHONY: all clean rebuild directories check-math check-rng test
