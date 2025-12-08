ifndef MATRIXLIBRARY
$(error MATRIXLIBRARY is not set, set it using export MATRIXLIBRARY=<path_to_tws_matrix>)
endif

CXX = g++
CPPFLAGS = -O0 -Wall -Wextra -pedantic -std=c++20

HEADERS=cg.hpp

all: test_cg q4

test_cg: test_cg.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ $< -I"$(MATRIXLIBRARY)"

q4: question4.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ $< -I"$(MATRIXLIBRARY)"

clean:
	rm -f test_cg *.o