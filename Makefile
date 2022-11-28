SRC := src/stencil.cxx
BIN := stencil

gnu: $(SRC)
	g++ -mcpu=native -Ofast -funroll-loops -finline-functions -fopenmp -g $? -o $(BIN)

arm: $(SRC)
	armclang++ -mcpu=native -Ofast -funroll-loops -finline-functions -fopenmp -g $? -o $(BIN)

nvc: $(SRC)
	nvc++ -march=neoverse-v1 -O4 -finline-functions -mp -g $? -o $(BIN)

itl: $(SRC)
	icpc -xHost -Ofast -funroll-loops -finline-functions -qopenmp $? -o $(BIN)

run: $(BIN)
	./stencil.sh

check: $(BIN)
	python3 scripts/speedup.py

maqao:
	~/maqao.aarch64.2.16.0/maqao.aarch64 oneview -R1 --config=cfg.lua -xp=maqao --replace
	tar cvf res.tar maqao/RESULTS/ 
