##!/bin/bash

#
# There is no cmake support for modules yet. This build script build with clang++
#

clang++ -std=c++2a -fmodules-ts --precompile math.cppm -o math.pcm              // 1
clang++ -std=c++2a -fmodules-ts  -c math.pcm -o math1.pcm.o                     // 2
clang++ -std=c++2a -fmodules-ts -c math.cpp -fmodule-file=math.pcm -o math.o    // 2
clang++ -std=c++2a -fmodules-ts -c main.cpp -fmodule-file=math.pcm -o main.o    // 3
clang++  math.pcm main.o math.o -o math                                         // 4