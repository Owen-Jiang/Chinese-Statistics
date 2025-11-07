#!/bin/bash
set -euo pipefail
mkdir -p data

function get_from_ucd_and_unzip() {
    curl -L -o "data/$1.zip" "https://www.unicode.org/Public/UCD/latest/ucd/${2:-$1}.zip"
    unzip "data/$1.zip" -d data/
    rm "data/$1.zip"
}

# Pull the unihan files in
echo "> Unihan"
get_from_ucd_and_unzip    Unihan
