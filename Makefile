SRC := src/stencil.cxx
BIN := stencil
COMMON_FLAGS := -Ofast -finline-functions -g -fno-omit-frame-pointer

gnu: $(SRC)
	g++ -march=native $(COMMON_FLAGS) -funroll-loops -fopenmp $? -o $(BIN)

llvm: $(SRC)
	clang++ -march=native $(COMMON_FLAGS) -funroll-loops -fopenmp $? -o $(BIN)

arm: $(SRC)
	armclang++ -mcpu=native $(COMMON_FLAGS) -funroll-loops -fopenmp $? -o $(BIN)

nvc: $(SRC)
	nvc++ -march=neoverse-v1+sve $(COMMON_FLAGS) -mp $? -o $(BIN)

intel: $(SRC)
	icpx -xHost $(COMMON_FLAGS) -mavx -funroll-loops -qopenmp -qmkl $? -o $(BIN)

run: $(BIN)
	@./stencil 100 100 100 5
	
check: $(BIN)
	@python3 scripts/speedup.py

prof:
	@maqao oneview -R1 --config=cfg.lua -xp=maqao --replace

