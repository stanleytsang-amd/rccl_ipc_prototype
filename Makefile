HIP_PATH?= $(wildcard /opt/rocm/hip)
MPI_PATH?= $(wildcard /home/WORKSPACE/statsang/ompi)

HIPCC=$(HIP_PATH)/bin/hipcc
CXX=$(HIPCC)
CXXFLAGS=-I$(MPI_PATH)/include -L$(MPI_PATH)/lib -lomp -lmpi -lrt -Ofast

PROTOTYPE = prototype
ALL = $(PROTOTYPE)

all: $(ALL)

$(TARGET): $(TARGET:.cpp)
	$(CXX) $(CXXFLAGS) -o $@ $^

.PHONY: clean
clean:
	rm -f *.o $(ALL)
