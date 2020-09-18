##!/bin/bash

#
# There is no cmake support for modules yet. This build script build with clang++
#

clang++ -std=c++2a -fmodules-ts --precompile module_math.cppm -o module_math.pcm
clang++ -std=c++2a -fmodules-ts  -c module_math.pcm -o math1.pcm.o
clang++ -std=c++2a -fmodules-ts -c module_math.cpp -fmodule-file=module_math.pcm -o module_math.o
clang++ -std=c++2a -fmodules-ts -c main.cxx -fmodule-file=module_math.pcm -o main.o
clang++ module_math.pcm main.o module_math.o -o main.out