ifndef MATRIXLIBRARY
$(error MATRIXLIBRARY is not set, set it using export MATRIXLIBRARY=<path_to_tws_matrix>)
endif

CXX=g++
CPPFLAGS=-std=c++20 -Wall -Wextra -pedantic

HEADERS=lu.hpp triangular_solve.hpp matmul.hpp

run_all_code: run_all_code.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ $<  -I"$(MATRIXLIBRARY)" -lblas -llapack -llapacke

clean:
	rm -f run_all_code *.o