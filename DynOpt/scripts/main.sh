#!/bin/bash

lrates=(0.001 0.002 0.004)

for lr in "${lrates[@]}"
do
	./subscript1.sh $lr &
done

wait
