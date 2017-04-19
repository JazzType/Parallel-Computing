# Parallel-Computing
Repo for CS336 - Parallel Computing Course Final Project

Companion blog: http://bookish-enigma.blogspot.in

## Project description

Parallelizing GNU comm

###### Optimizations applied
- AVX2 and SSE4.1
- Multithreading (pthreads)

## Try it out

1. Clone the repo
2. Build using one of the following commands:
- `$ make all` - Build both threaded and non-threaded binaries 
- `$ make simd` - Build only threaded binary
- `$ make thread` - Build only non-threaded binary

Running `make` will build both binaries by default.

#### Running tests

Navigate to /tests folder, and run `./run-tests.sh`. It will output all three times for each input file, for all three binaries.
