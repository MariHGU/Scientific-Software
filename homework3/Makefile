ifndef MATRIXLIBRARY
$(error MATRIXLIBRARY is not set, set it using export MATRIXLIBRARY=<path_to_tws_matrix>)
endif

CXX=g++
CPPFLAGS= -std=c++20 -Wall -Wextra -pedantic

HEADERS=lu.hpp triangular_solve.hpp matmul.hpp

all: run_all_code test_lu time_lu

run_all_code: CPPFLAGS = -std=c++20 -Wall -Wextra -pedantic
run_all_code: run_all_code.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ $<  -I"$(MATRIXLIBRARY)" -lblas -llapack

test_lu: CPPFLAGS = -O0 -g -std=c++20 -Wall -Wextra -pedantic
test_lu: test_lu.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ $< -I"$(MATRIXLIBRARY)" -lblas -llapack

time_lu: CPPFLAGS = -O3 -DNDEBUG -march=native -std=c++20
time_lu: time_lu.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ $< -I"$(MATRIXLIBRARY)" -lblas -llapack

clean:
	rm -f run_all_code test_lu time_lu *.o