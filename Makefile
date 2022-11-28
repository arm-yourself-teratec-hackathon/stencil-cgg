SRC := src/stencil.cxx
BIN := stencil

gnu: $(SRC)
	g++ -mcpu=native -Ofast -funroll-loops -finline-functions -fopenmp -g $? -o $(BIN)

arm: $(SRC)
	armclang++ -mcpu=native -Ofast -funroll-loops -finline-functions -fopenmp -g $? -o $(BIN)

nvc: $(SRC)
	nvc++ -march=neoverse-v1 -O4 -funroll-loops -finline-functions -fopenmp -g $? -o $(BIN)

itl: $(SRC)
	icpc -xHost -Ofast -funroll-loops -finline-functions -qopenmp $? -o $(BIN)

run: $(BIN)
	./stencil.sh
	
check: $(BIN)
	python3 scripts/speedup.py

