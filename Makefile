
NVCC = nvcc
# Optimization and source-mapping for Nsight Compute; do NOT use -G here
NVCC_FLAGS = -O3 -lineinfo -Xcompiler -Wall

# Default target architecture (compute for PTX, sm for native SASS).
# Override by calling, e.g. `make NVCC_ARCH=86`
NVCC_ARCH ?= 86
# Produce both native SASS (sm_XX) and PTX (compute_XX)
NVCC_GENCODE = -gencode arch=compute_$(NVCC_ARCH),code=sm_$(NVCC_ARCH) \
			  -gencode arch=compute_$(NVCC_ARCH),code=compute_$(NVCC_ARCH)

# Include paths
INCLUDE_FLAGS = -I. -I./include

# Folders
DRIVERS_DIR = drivers
MHA_KERNELS_DIR = mha_kernels
INPUTS_DIR = inputs
UTILS_DIR = utils
TOOLS_DIR = tools
BIN_DIR = bin

# Targets
TARGET = $(BIN_DIR)/main

# Source files
SRCS = $(DRIVERS_DIR)/main.cu \
       $(MHA_KERNELS_DIR)/unfused.cu \
       $(MHA_KERNELS_DIR)/fa.cu \
       $(MHA_KERNELS_DIR)/fa_warps.cu \
       $(MHA_KERNELS_DIR)/fa_tc.cu \
       $(MHA_KERNELS_DIR)/fa_int8.cu \
       $(INPUTS_DIR)/data.cu \
       $(UTILS_DIR)/verify.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODE) $(INCLUDE_FLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -rf $(BIN_DIR)
	rm -f $(MHA_KERNELS_DIR)/*.o $(INPUTS_DIR)/*.o $(DRIVERS_DIR)/*.o $(UTILS_DIR)/*.o