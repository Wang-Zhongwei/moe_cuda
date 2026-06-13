NVCC    ?= nvcc
NVFLAGS ?= -O2 -std=c++14 -Iinclude

SRC := $(wildcard src/*/*.cu) app/main.cu
OBJ := $(SRC:.cu=.o)
BIN := moe

all: $(BIN)

$(BIN): $(OBJ)
	$(NVCC) $(NVFLAGS) $^ -o $@

%.o: %.cu
	$(NVCC) $(NVFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: all clean
