
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

# Kernel selection (default: unfused)
# Available kernels: unfused, fa, fa_warps, fa_tc, fa_int8
# Usage: make KERNEL=fa_int8 NVCC_ARCH=80
KERNEL ?= unfused

# Targets and sources
TARGET = $(BIN_DIR)/profile_$(KERNEL)

# Base source files (always included)
BASE_SRCS = $(DRIVERS_DIR)/main.cu \
            $(INPUTS_DIR)/data.cu \
            $(UTILS_DIR)/verify.cu

# Kernel-specific source file
KERNEL_SRC = $(MHA_KERNELS_DIR)/$(KERNEL).cu

# All sources combined
SRCS = $(BASE_SRCS) $(KERNEL_SRC)

.PHONY: all clean

all: $(TARGET)
	@echo "Built: $(TARGET) (kernel=$(KERNEL), arch=sm_$(NVCC_ARCH))"

$(TARGET): $(SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODE) $(INCLUDE_FLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -rf $(BIN_DIR)
	rm -f $(MHA_KERNELS_DIR)/*.o $(INPUTS_DIR)/*.o $(DRIVERS_DIR)/*.o $(UTILS_DIR)/*.o