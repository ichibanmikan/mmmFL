#!/bin/bash

for node_id in {11..21}
do
    echo "start $node_id"
    CUDA_VISIBLE_DEVICES=2 python client.py --node_id $node_id &
done

wait
echo "finish"