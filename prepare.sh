#!/bin/bash

a="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$a/server"

mv "$a/Test" "$a/server/test_datasets"

cd "$a/Train" || exit

for train_subdir in */; do
    dir_name="${train_subdir%/}"
    if [ -d "$a/client/$dir_name" ]; then
        cp -r "$a/Train/$dir_name/"* "$a/client/$dir_name/"
    fi
done
