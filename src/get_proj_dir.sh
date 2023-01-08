#!/bin/bash

set -e

get_proj_root () {
  # requires that file .root.txt is located in top level of project directory.
  # if you don't want to do this, hard-code or input the project dir as an arg.
  local DIR=$(pwd)
  while [ ! -z "$DIR" ] && [ ! -f "$DIR/.root.txt" ]; do DIR="${DIR%\/*}"; done
  if [ -z "$DIR" ]; then script=`basename "$0"` && echo "ERROR: could not determine project directory in script $script. Exiting." >&2 && exit 1; fi
  echo "$DIR"
}

