#!/usr/bin/env bash

folder=$1
for f in $(find $folder -name '*.m4a'); do ffmpeg -y -i "$f" -ac 1 -ar 16000 "${f%.m4a}.wav" -loglevel quiet; done
