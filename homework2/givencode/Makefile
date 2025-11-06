ifndef MATRIXLIBRARY
$(error MATRIXLIBRARY is not set, set it using export MATRIXLIBRARY=<path_to_tws_matrix>)
endif

CXX=g++
CPPFLAGS=-std=c++20 -Wall -Wextra -pedantic

HEADERS=matmul.hpp matmul_kernel.hpp

all: run_all_matmul

run_all_matmul: run_all_matmul.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ $< -I$(MATRIXLIBRARY)

clean:
	rm -f run_all_matmul *.o