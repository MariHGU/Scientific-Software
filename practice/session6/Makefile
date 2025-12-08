CXX=g++ 
CXXFLAGS=-O0 -std=c++20 -Wall -Wextra -I"$(MATRIXLIBRARY)"

HEADERS=c_add.h fortran_add.h

c_add.o: c_add.c $(HEADERS)
	gcc -c $< -o $@

fortran_add.o: fortran_add.f90 fortran_add.h
	gfortran -o $@ -c $<

main.o: main.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ -c $< 

main: main.o c_add.o
	$(CXX) -o $@ $^

main_f.o: main_f.cpp fortran_add.h
	$(CXX) $(CXXFLAGS) -o $@ -c $< 

main_f: main_f.o fortran_add.o
	$(CXX) -o $@ $^ -lgfortran

clean:
	rm -f main main_f *.o