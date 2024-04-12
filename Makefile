# Compiler and flags (for Clang++ with Intel LLVM SYCL on Nvidia GPU)
CXX = clang++
CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 -Wno-deprecated-declarations #-std=c++17

# Source files and the target executable name
SRCS = R_spanner_helper.cxx printer.cxx R_spanner_kernels.cxx R_spanner.cxx
TARGET = op_R_spannerNew

# Default rule to build the target
all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Rule to clean generated files
clean:
	rm -f $(TARGET)
