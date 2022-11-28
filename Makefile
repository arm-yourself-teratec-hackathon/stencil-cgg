SRC := src/stencil.cxx
BIN := stencil

gnu: $(SRC)
	g++ -mcpu=native -Ofast -funroll-loops -finline-functions -fopenmp -g $? -o $(BIN)

arm: $(SRC)
	armclang++ -mcpu=native -Ofast -funroll-loops -finline-functions -fopenmp -g $? -o $(BIN)

nvc: $(SRC)
	nvc++ -march=neoverse-v1 -O4 -funroll-loops -finline-functions -fopenmp -g $? -o $(BIN)

itl: $(SRC)
	icpc -xHost -mavx -Ofast -funroll-loops -finline-functions -qopenmp -qmkl -g $? -o $(BIN)

run: $(BIN)
	./stencil 100 100 100 5
	
check: $(BIN)
	python3 scripts/speedup.py

prof:
	maqao oneview -R1 --config=cfg.lua -xp=maqao --replace

