#!/bin/bash

set -e

script_dir=`dirname $0`

"$script_dir/0.0_download_genome.sh"
"$script_dir/0.1_download_all_data.sh"
"$script_dir/0.2_download_histone_marks.sh"
