

CXX = g++
CPPFLAGS = -O0 -Wall - Wextra -pedantic -std=c++20

HEADERS=cg.hpp

test_cg: test_cg.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ $<

clean:
	rm -f test_cg *.o