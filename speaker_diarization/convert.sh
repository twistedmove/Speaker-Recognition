#!/usr/bin/env bash

for f in $(find /hdd/VoxCeleb2/vox2_dev_dataset/dev/aac/ -name '*.m4a'); do ffmpeg -y -i "$f" -ac 1 -ar 16000 "${f%.m4a}.wav" -loglevel quiet; done