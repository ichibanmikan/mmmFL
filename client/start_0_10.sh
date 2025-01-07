#!/bin/bash

for node_id in {0..10}
do
    echo "start $node_id"
    CUDA_VISIBLE_DEVICES=1 python client.py --node_id $node_id &
done

wait
echo "finish"