CXX=g++ 
CXXFLAGS=-O0 -std=c++20 -Wall -Wextra

HEADERS=c_add.h

c_add: c_add.c $(HEADERS)
	gcc -c $< -o $@

main.o: main.cpp $(HEADERS)
	$(CXX) $(CPPFLAGS) -o $@ -c $< -I"$(MATRIXLIBRARY)"

main: main.o c_add.o
	$(CXX) -o $@ $^

clean:
	rm -f main *.o