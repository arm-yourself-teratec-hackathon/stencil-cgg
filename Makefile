SRC := stencil.cxx
BIN := stencil

gnu: $(SRC)
	g++ -mcpu=native -Ofast -funroll-loops -finline-functions $? -o $(BIN)

arm: $(SRC)
	armclang++ -mcpu=native -Ofast -funroll-loops -finline-functions $? -o $(BIN)

nvc: $(SRC)
	nvc++ -march=neoverse-v1 -O4 -funroll-loops -finline-functions $? -o $(BIN)

run: $(BIN)
	./stencil.sh

