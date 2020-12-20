# english compiler output
LANG=C

VERSION=10.0
MLINKDIR = /usr/local/Wolfram/Mathematica/${VERSION}/SystemFiles/Links/MathLink/DeveloperKit
SYS = Linux-x86-64
CADDSDIR = ${MLINKDIR}/${SYS}/CompilerAdditions
EXTRA_CFLAGS=-m64 -march=native

INCDIR = -I${CADDSDIR} -I../include
LIBDIR = -L${CADDSDIR} -L../lib

MPREP = ${CADDSDIR}/mprep
RM = rm

CC = /usr/bin/gcc
CXX = /usr/bin/g++

CUDA_ROOT = /usr/local/cuda

BINARIES = laxcudamlink laxcudamlinkstatic

LDLIBS_COMMON = -lML64i4 -luuid -lm -lpthread -lrt -lstdc++ -lboost_timer -lboost_chrono -lboost_system -lboost_thread

LDLIBS_SHARED = -llaxcuda
LDLIBS_SHARED += $(LDLIBS_COMMON)
LDLIBS_STATIC = -l:liblaxcuda.a -lcudart -lgomp
LDLIBS_STATIC += $(LDLIBS_COMMON)

all : $(BINARIES)

install : $(BINARIES)
	cp -t ~/lib/laxcuda $(BINARIES)

laxcudamlink : laxcudamlinktm.o
	${CC} -Wl ${EXTRA_CFLAGS} -o $@  laxcudamlinktm.o ${LIBDIR} $(LDLIBS_SHARED)

laxcudamlinkstatic : laxcudamlinktm.o
	${CC} -Wl ${EXTRA_CFLAGS} -o $@  laxcudamlinktm.o ${LIBDIR} -L$(CUDA_ROOT)/lib64 $(LDLIBS_STATIC)

.c.o :
	${CC} -Wall -c ${EXTRA_CFLAGS} ${INCDIR} $<

laxcudamlinktm.c : laxcudamlink.tm
	${MPREP} $? -o $@

clean :
	${RM} -rf *.o *tm.c $(BINARIES)
