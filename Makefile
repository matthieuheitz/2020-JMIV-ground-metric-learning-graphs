CFLAGS=-Wall -fPIC -fopenmp
CPPFLAGS=-Wall -fPIC -fopenmp
LDFLAGS=-shared

dynamic_c: ckernels.c
	gcc $(CFLAGS) -c ckernels.c
	gcc $(LDFLAGS) ckernels.o -o libckernels.so

test_c: ckernels.c
	gcc $(CFLAGS) ckernels.c -o test_ckernel -lm

all: dynamic_c

tests: test_c

.PHONY: clean

clean:
	rm -rf *.o

veryclean: clean
	rm -rf *.so



