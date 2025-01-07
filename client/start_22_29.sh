#!/bin/bash

for node_id in {22..29}
do
    echo "start $node_id"
    CUDA_VISIBLE_DEVICES=0 python client.py --node_id $node_id &
done

wait
echo "finish"