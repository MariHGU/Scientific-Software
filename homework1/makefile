# # --- Makefile ---

CXX := g++

# Base flags
BASE_CXXFLAGS := -std=c++20 -Wall -Wextra -I "$(MATRIXLIBRARY)"

# Default: release build for performance
CXXFLAGS := $(BASE_CXXFLAGS) -O3 -march=native -DNDEBUG

# Source files
SRCS := simulate_system.cpp compute_gradient.cpp optimal_control.cpp
OBJS := $(SRCS:.cpp=.o)
TARGET := simulate_system compute_gradient optimal_control

# Default target
all: $(TARGET)

# Debug build
debug: CXXFLAGS := $(BASE_CXXFLAGS) -g -O0
debug: $(TARGET)

# Release build
release: CXXFLAGS := $(BASE_CXXFLAGS) -O3 -march=native -DNDEBUG
release: $(TARGET)

# Link .o -> executable
$(TARGET): $(OBJS)
	$(CXX) $^ -o $@

# Compile .cpp -> .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJS) $(TARGET)
