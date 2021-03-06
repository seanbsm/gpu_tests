# the compiler to use
CPP  := g++
NVCC := nvcc
FORTC := gfortran 

# -no-pie -g -pg for gprof

TARGET := run

# Source, object, and dependency files for c++, cuda, and fortran
SRC_FILES := $(shell find ./ -regex [^\#]*\\.cpp$)
OBJ_FILES := $(SRC_FILES:.cpp=.o)
DEP_FILES := $(SRC_FILES:.cpp=.d)

SRCCUDA_FILES := $(shell find ./ -regex [^\#]*\\.cu$)
OBJCUDA_FILES := $(SRCCUDA_FILES:%.cu=%.o)
DEPCUDA_FILES := $(SRCCUDA_FILES:.cu=.d)

SRCFORT_FILES := $(shell find ./ -regex [^\#]*\\.f90$)
OBJFORT_FILES := $(SRCFORT_FILES:.f90=.o)

# Flags
CUDAFLAGS := -O2 -Xcompiler -fPIC
CPPFLAGS :=  -Wall -std=c++17 -O2

# Libraries
#~ MAGMALIBS := -L/usr/local/magma/lib -L/lib64 -L/home/sean/intel/compilers_and_libraries_2019.4.243/linux/mkl/lib/intel64 -lmagma_sparse -lmagma -fopenmp -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -lgfortran -lcusparse -lcudadevrt
CUDALIBS := -lcuda -lcudart -lcublas -lcusolver
LDLIBS := -L$(MKLROOT)/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl -lgfortran 


# Symbol explenations
# @ = recipe name (so $(TARGET) in this case)
# ^ = all dependencies (so ALL $(OBJ_FILES))
# < = first dependency (so first of OBJ_FILES)

# Include the .d files
-include $(DEP_FILES) $(DEPCUDA_FILES)
#-include $(DEPCUDA_FILES)

# Make executable
$(TARGET): $(OBJ_FILES) $(OBJCUDA_FILES) $(OBJFORT_FILES)
	$(CPP) $^ -o $@ $(LDFLAGS) $(LDLIBS) $(CUDALIBS)

# Make object files
%.o: %.cpp makefile
	$(CPP) $(CPPFLAGS) -MMD -MP -c $< -o $@

%.o: %.cu makefile
	$(NVCC) $(CUDAFLAGS) -c $< -o $@
	
%.o: %.f90 makefile
	$(FORTC) $(FORTFLAGS) -c $< -o $@


.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJ_FILES) $(OBJCUDA_FILES) $(OBJFORT_FILES) $(DEP_FILES) *.o *.mod *.exe
