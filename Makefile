CUDA_TARGET += --generate-code arch=compute_20,code=sm_20
#CUDA_TARGET += --generate-code arch=compute_20,code=sm_21 
CUDA_TARGET += --generate-code arch=compute_30,code=sm_30
CUDA_TARGET += --generate-code arch=compute_35,code=sm_35
#CUDA_TARGET += --generate-code arch=compute_20,code=compute_20 
#CUDA_TARGET += --generate-code arch=compute_30,code=compute_30 
#CUDA_TARGET += --generate-code arch=compute_35,code=compute_35

CUDA_ROOT = /usr/local/cuda
THRUST_ROOT = external_libs/thrust-1.7.0
#BOOST_ROOT = /home/slawa/software/boost/install/
ODEINT_ROOT = external_libs/odeint-v2

LIB_NAME = laxcuda
LIB_DIRECTORY = lib
INCLUDE_DIRECTORY = include
TEST_BIN_DIRECTORY = bin
OBJECT_DIRECTORY = compile

DEBUG_FLAGS = -DINFO_PRINT_VERBOSE

LIB_SHARED = $(LIB_DIRECTORY)/lib$(LIB_NAME).so
LIB_STATIC = $(LIB_DIRECTORY)/lib$(LIB_NAME).a

# english compiler output
LANG=C

CC = gcc
CXX = g++
NVCC = $(CUDA_ROOT)/bin/nvcc

INCLUDES_COMMON = -I$(CUDA_ROOT)/include  
INCLUDES_LIB = -I./src -I$(THRUST_ROOT) -I$(ODEINT_ROOT) $(INCLUDES_COMMON)
INCLUDES_TEST = -I$(INCLUDE_DIRECTORY) $(INCLUDES_COMMON) -I/usr/local/cuda/samples/common/inc

CCFLAGS_LIB = -march=native -Wall -O3 $(INCLUDES_LIB) -Xcompiler -fopenmp -Xcompiler -fpic 
NVCCFLAGS_LIB = -Xcompiler -march=native -Xcompiler -Wall -O3 $(INCLUDES_LIB) $(CUDA_TARGET) -Xcompiler -fopenmp -Xcompiler -fpic

CCFLAGS_LIB += $(DEBUG_FLAGS)
NVCCFLAGS_LIB += $(DEBUG_FLAGS)

CCFLAGS_TEST = -march=native -Wall -O3 $(INCLUDES_TEST)

LDFLAGS = -L$(LIB_DIRECTORY) -L$(CUDA_ROOT)/lib64
#LDFLAGS += -L$(BOOST_ROOT)/lib

LDLIBS_COMMON = -lboost_timer -lboost_chrono -lboost_system -lboost_thread

LDLIBS_SHARED = -l$(LIB_NAME) -lcudart -lgomp
LDLIBS_SHARED += $(LDLIBS_COMMON)
LDLIBS_SO = -lcudart -lgomp
LDLIBS_SO += $(LDLIBS_COMMON)
LDLIBS_STATIC = -l:$(LIB_STATIC) -lcudart -lgomp
LDLIBS_STATIC += $(LDLIBS_COMMON)

CC_SRCS_LIB = 
NVCC_SRCS_LIB = src/xlab/surf/cuda/laxode/laxcuda.cu

LIB_HEADERS = src/xlab/surf/cuda/laxode/laxdefinitions.h\
              src/xlab/surf/cuda/laxode/laxcudaerrors.h\
			  src/xlab/surf/cuda/laxode/laxcuda.hpp\
			  src/xlab/surf/cuda/laxode/laxcuda.h

CC_SRCS_TEST = src/test/LaxODETest.cpp src/test/LaxODETestCInterface.c

CC_OBJS_LIB := $(sort $(patsubst %.cpp,$(OBJECT_DIRECTORY)/%.o,$(patsubst %.c,$(OBJECT_DIRECTORY)/%.o,$(notdir $(CC_SRCS_LIB)))))
NVCC_OBJS_LIB := $(sort $(patsubst %.cu,$(OBJECT_DIRECTORY)/%.o,$(notdir $(NVCC_SRCS_LIB))))

CC_OBJS_TEST := $(sort $(patsubst %.cpp,$(OBJECT_DIRECTORY)/%.o,$(patsubst %.c,$(OBJECT_DIRECTORY)/%.o,$(notdir $(CC_SRCS_TEST)))))

CC_TARGETS_TEST := $(sort $(patsubst %.o,$(TEST_BIN_DIRECTORY)/%,$(notdir $(CC_OBJS_TEST))))
CC_TARGETS_TEST_STATIC := $(sort $(patsubst %.o,$(TEST_BIN_DIRECTORY)/%.static,$(notdir $(CC_OBJS_TEST))))

LIB_HEADERS_TARGETS := $(sort $(patsubst %,$(INCLUDE_DIRECTORY)/%,$(notdir $(LIB_HEADERS))))


#test :
#	@echo "$(LIB_HEADERS_TARGETS)"

all : lib linktests mathematica

lib : $(LIB_SHARED) $(LIB_STATIC) $(LIB_HEADERS_TARGETS)

compilelib : $(NVCC_OBJS_LIB) $(CC_OBJS_LIB)

copylibheaders : $(LIB_HEADERS_TARGETS)

compiletests : $(CC_OBJS_TEST)

linktests : $(CC_TARGETS_TEST) $(CC_TARGETS_TEST_STATIC)

runtests : linktests
	@$(foreach test,$(CC_TARGETS_TEST),LD_LIBRARY_PATH=$(LIB_DIRECTORY) $(test);)

runfirsttest : linktests
	LD_LIBRARY_PATH=$(LIB_DIRECTORY) $(word 1,$(CC_TARGETS_TEST)) --grid_device=1

runfirsttestcuda : linktests
	LD_LIBRARY_PATH=$(LIB_DIRECTORY) $(word 1,$(CC_TARGETS_TEST)) --grid_device=5


writecsv : linktests
	LD_LIBRARY_PATH=$(LIB_DIRECTORY) bin/LaxODETest --Nu=101 --Nv=101 --csv_output=integrationresults.csv

clean :
	rm -rf $(OBJECT_DIRECTORY)/* $(LIB_DIRECTORY)/* $(INCLUDE_DIRECTORY)/* $(TEST_BIN_DIRECTORY)/* LaxODE.tar.gz
	cd mathematica; make clean

package : linktests
	rm -rf LaxODE.tar.gz
	tar czvf LaxODE.tar.gz Makefile src/ $(OBJECT_DIRECTORY)/ $(LIB_DIRECTORY)/ $(INCLUDE_DIRECTORY)/ $(TEST_BIN_DIRECTORY)/ external_libs/ mathematica/ integrationresults.csv

install : lib
	cp -t ~/lib/laxcuda/ $(LIB_HEADERS_TARGETS) $(LIB_STATIC) $(LIB_SHARED)
	cd mathematica; make install

mathematica : lib
	cd mathematica; make all

# build library

$(LIB_STATIC) : $(NVCC_OBJS_LIB) $(CC_OBJS_LIB) $(LIB_HEADERS_TARGETS)
	@echo
	@mkdir -p $(LIB_DIRECTORY)
	@echo "building $(LIB_STATIC)"
	rm -rf $(LIB_STATIC)
	ar rucs $(LIB_STATIC) $(CC_OBJS_LIB) $(NVCC_OBJS_LIB)
	@echo "built $(LIB_STATIC)"
	@echo

$(LIB_SHARED) : $(NVCC_OBJS_LIB) $(CC_OBJS_LIB) $(LIB_HEADERS_TARGETS)
	@echo
	@mkdir -p $(LIB_DIRECTORY)
	@echo "building $(LIB_SHARED)"
	$(CC) -shared -Wl,-soname,lib$(LIB_NAME).so -o $(LIB_SHARED) $(CC_OBJS_LIB) $(NVCC_OBJS_LIB) $(LDFLAGS) $(LDLIBS_SO)
	@echo "built $(LIB_SHARED)"
	@echo


# rules for linking tests

$(TEST_BIN_DIRECTORY)/% : $(OBJECT_DIRECTORY)/%.o $(LIB_SHARED)
	@echo
	@echo "$(notdir $(CC)) is linking '$@'"
	@mkdir -p $(@D)
	$(CC) -Wl -o $@ $< $(LDFLAGS) $(LDLIBS_SHARED)
	@echo "$(notdir $(CC)) built '$@'"
	@echo

$(TEST_BIN_DIRECTORY)/%.static : $(OBJECT_DIRECTORY)/%.o $(LIB_STATIC)
	@echo
	@echo "$(notdir $(CC)) is linking '$@'"
	@mkdir -p $(@D)
	$(CC) -Wl -o $@ $< $(LDFLAGS) $(LDLIBS_STATIC)
	@echo "$(notdir $(CC)) built '$@'"
	@echo

define copy_header_rule
$(INCLUDE_DIRECTORY)/$(notdir $(1)) : $(1)
	@echo
	@mkdir -p $$(@D)
	@echo "copy $$(@F) to $(INCLUDE_DIRECTORY)"
	cp -a $$< $(INCLUDE_DIRECTORY)/
	@echo
endef

$(foreach header,$(sort $(LIB_HEADERS)),$(eval $(call copy_header_rule,$(header))))

# generate compile rules with automatic dependencies 

define cc_lib_rule
$(OBJECT_DIRECTORY)/%.o : $(1)%.c
	@echo
	@echo "$(notdir $$(CC)) is compiling '$$<'"
	@mkdir -p $$(@D)
	$(CC) $(CCFLAGS_LIB) -o $$@ -c $$<
	$(CC) $(CCFLAGS_LIB) -M $$< > $$(@:%.o=%.d)
	@mv -f $$(@:%.o=%.d) $$(@:%.o=%.d).tmp
	@sed -e 's|.*:|$$@:|' < $$(@:%.o=%.d).tmp > $$(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$$$//' < $$(@:%.o=%.d).tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$$$/:/' >> $$(@:%.o=%.d)
	@rm -f $$(@:%.o=%.d).tmp
	@echo "$(notdir $(CC)) finished compiling '$$<'"
	@echo
endef

$(foreach directory,$(sort $(dir $(CC_SRCS_LIB))),$(eval $(call cc_lib_rule,$(directory))))

define cpp_lib_rule
$(OBJECT_DIRECTORY)/%.o : $(1)%.cpp
	@echo
	@echo "$(notdir $(CXX)) is compiling '$$<'"
	@mkdir -p $$(@D)
	$(CXX) $(CCFLAGS_LIB) -o $$@ -c $$<
	$(CXX) $(CCFLAGS_LIB) -M $$< > $$(@:%.o=%.d)
	@mv -f $$(@:%.o=%.d) $$(@:%.o=%.d).tmp
	@sed -e 's|.*:|$$@:|' < $$(@:%.o=%.d).tmp > $$(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$$$//' < $$(@:%.o=%.d).tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$$$/:/' >> $$(@:%.o=%.d)
	@rm -f $$(@:%.o=%.d).tmp
	@echo "$(notdir $(CXX)) finished compiling '$$<'"
	@echo
endef

$(foreach directory,$(sort $(dir $(CC_SRCS_LIB))),$(eval $(call cpp_lib_rule,$(directory))))

define nvcc_lib_rule
$(OBJECT_DIRECTORY)/%.o : $(1)%.cu
	@echo
	@echo "$(notdir $(NVCC)) is compiling '$$<'"
	@mkdir -p $$(@D)
	$(NVCC) $(NVCCFLAGS_LIB) -o $$@ -c $$<
	$(NVCC) $(NVCCFLAGS_LIB) -M $$< > $$(@:%.o=%.d)
	@mv -f $$(@:%.o=%.d) $$(@:%.o=%.d).tmp
	@sed -e 's|.*:|$$@:|' < $$(@:%.o=%.d).tmp > $$(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$$$//' < $$(@:%.o=%.d).tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$$$/:/' >> $$(@:%.o=%.d)
	@rm -f $$(@:%.o=%.d).tmp
	@echo "$(notdir $(NVCC)) finished compiling '$$<'"
	@echo
endef

$(foreach directory,$(sort $(dir $(NVCC_SRCS_LIB))),$(eval $(call nvcc_lib_rule,$(directory))))

define cc_test_rule
$(OBJECT_DIRECTORY)/%.o : $(1)%.c $(LIB_HEADERS_TARGETS)
	@echo
	@echo "$(notdir $$(CC)) is compiling '$$<'"
	@mkdir -p $$(@D)
	$(CC) $(CCFLAGS_TEST) -o $$@ -c $$<
	$(CC) $(CCFLAGS_TEST) -M $$< > $$(@:%.o=%.d)
	@mv -f $$(@:%.o=%.d) $$(@:%.o=%.d).tmp
	@sed -e 's|.*:|$$@:|' < $$(@:%.o=%.d).tmp > $$(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$$$//' < $$(@:%.o=%.d).tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$$$/:/' >> $$(@:%.o=%.d)
	@rm -f $$(@:%.o=%.d).tmp
	@echo "$(notdir $(CC)) finished compiling '$$<'"
	@echo
endef

$(foreach directory,$(sort $(dir $(CC_SRCS_TEST))),$(eval $(call cc_test_rule,$(directory))))

define cpp_test_rule
$(OBJECT_DIRECTORY)/%.o : $(1)%.cpp $(LIB_HEADERS_TARGETS)
	@echo
	@echo "$(notdir $(CXX)) is compiling '$$<'"
	@mkdir -p $$(@D)
	$(CXX) $(CCFLAGS_TEST) -o $$@ -c $$<
	$(CXX) $(CCFLAGS_TEST) -M $$< > $$(@:%.o=%.d)
	@mv -f $$(@:%.o=%.d) $$(@:%.o=%.d).tmp
	@sed -e 's|.*:|$$@:|' < $$(@:%.o=%.d).tmp > $$(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$$$//' < $$(@:%.o=%.d).tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$$$/:/' >> $$(@:%.o=%.d)
	@rm -f $$(@:%.o=%.d).tmp
	@echo "$(notdir $(CXX)) finished compiling '$$<'"
	@echo
endef

$(foreach directory,$(sort $(dir $(CC_SRCS_TEST))),$(eval $(call cpp_test_rule,$(directory))))

# pull in dependency info for *existing* .o files
-include $(CC_OBJS_LIB:.o=.d)
-include $(NVCC_OBJS_LIB:.o=.d)
-include $(CC_OBJS_TEST:.o=.d)

.PHONY: all clean lib copylibheaders compilelib compiletests linktests runtests runfirsttest runfirsttestcuda package writecsv mathematica install
