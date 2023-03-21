#!/bin/bash

make > /dev/null

printf "\033[1mG3E:\033[0m\n"
for i in $(seq 1 10); do
  ./stencil > output/test-e
  printf "  Run %2d\t" $i
  python3 scripts/assert.py -p official -o output/test-e | grep 'faster' | awk '{print $3"x"}'
done

printf "\n\033[1mG3:\033[0m\n"
for i in $(seq 1 10); do
  ./stencil-g3 > output/test-c
  printf "  Run %2d\t" $i
  python3 scripts/assert.py -p official -o output/test-c | grep 'faster' | awk '{print $3"x"}'
done
