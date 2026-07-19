CXX = mpicxx

CXXFLAGS = -O3 -std=c++20 -fopenmp -IInclude -I/usr/include/eigen3

SRC = $(wildcard Src/*.cpp)

TARGET = main

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)

NPROCS ?= 4

run: 
	mpirun -np $(NPROCS) ./$(TARGET)