#!/bin/bash

rm DataReader/training_list.py
touch DataReader/training_list.py

echo "classcounts = [begin_marker" >> training_list.py
for dr in $(ls train); do
    echo "[\""${dr}"\","$(ls ./train/${dr}/* | wc -l)"]," >> training_list.py
done
echo "endfile_marker]" >> training_list.py
gsed -i ':a;N;$!ba;s/begin_marker\n//g' training_list.py
gsed -i ':a;N;$!ba;s/,\nendfile_marker//g' training_list.py
