#!/bin/bash
# stats table rows with the CPU-time column stripped
awk '/^Step +CPU/{on=1;next} /^Loop time/{on=0} on && $1 ~ /^[0-9]+$/ { $2=""; print }' "$1"
