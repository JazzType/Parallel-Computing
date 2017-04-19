all: pcomm_threaded.c pcomm.c
	gcc pcomm.c -o pcomm_threaded -mavx2 -msse4.1 -pthread
	gcc pcomm.c -o pcomm -mavx2 -msse4.1

thread: pcomm_threaded.c
	gcc pcomm_threaded.c -o pcomm_threaded -mavx2 -msse4.1 -pthread

simd: pcomm.c
	gcc pcomm.c -o pcomm -mavx2 -msse4.1