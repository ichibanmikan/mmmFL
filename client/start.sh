#!/bin/bash

for node_id in {0..29}
do
    echo "start $node_id"
    python client.py --node_id $node_id &
done

wait
echo "finish"