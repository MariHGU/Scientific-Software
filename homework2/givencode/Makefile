ifndef MATRIXLIBRARY
$(error MATRIXLIBRARY is not set, set it using export MATRIXLIBRARY=<path_to_tws_matrix>)
endif

CXX=g++ -O3 -DNDEBUG -march=native
CPPFLAGS=-std=c++20
BASE_CXXFLAGS := -std=c++20 -I "$(MATRIXLIBRARY)"

HEADERS=matmul.hpp matmul_kernel.hpp

all: run_all_matmul compare_matricies time_matmul block_test

run_all_matmul: run_all_matmul.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ $< -I "$(MATRIXLIBRARY)"

compare_matricies: compare_matricies.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ $< -I "$(MATRIXLIBRARY)"

time_matmul: time_matmul.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ $< -I "$(MATRIXLIBRARY)"

block_test: block_test.cpp 
	$(CXX) $(CPPFLAGS) -o $@ $< -I "$(MATRIXLIBRARY)"

clean:
	rm -f run_all_matmul compare_matricies time_matmul block_test *.o