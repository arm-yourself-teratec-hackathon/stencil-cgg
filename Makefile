SRC := src/stencil.cxx
BIN := stencil
COMMON_FLAGS := -Ofast -finline-functions -g -fno-omit-frame-pointer
LDFLAGS := -larmpl

all: $(BIN)

$(BIN): $(SRC)
	g++ -march=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -ftree-loop-vectorize -fopenmp $? -o $(BIN) $(LDFLAGS)

llvm: $(SRC)
	clang++ -march=native $(COMMON_FLAGS) -funroll-loops -fopenmp $? -o $(BIN) $(LDFLAGS)

arm: $(SRC)
	armclang++ -mcpu=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN) $(LDFLAGS)

nvc: $(SRC)
	nvc++ -march=neoverse-v1 $(COMMON_FLAGS) -mp $? -o $(BIN) $(LDFLAGS)

intel: $(SRC)
	icpx -xHost $(COMMON_FLAGS) -mavx -funroll-loops -qopenmp -qmkl $? -o $(BIN)

run: $(BIN)
	@./stencil 100 100 100 5
	
check: $(BIN)
	@python3 scripts/assert.py $(ARGS)

prof:
	@maqao oneview -R1 --config=cfg.lua -xp=maqao_arm --replace

clean:
	@rm -rf $(BIN) slurm-jobs/* maqao-arm*
