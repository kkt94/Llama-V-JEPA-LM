#!/bin/bash
# usage: transcode_one.sh <in.mp4> <out.mp4> <fps|native> <gop>
# Short side -> 384 (aspect kept, no crop), H.264 CRF 23, no audio. Writes .part then renames, so reruns skip finished files.
# x264 preset ultrafast (was veryfast; 1.5x, larger files); EgoDex (1080p MPEG-4 part 2) decoded at half resolution
# (-lowres 1: 960x540, still above the 384 output) - 7.3 s -> 3.2 s per 39 s clip
in=$1; out=$2; fps=$3; gop=$4; FF=/data/llama_vjepa_lm/tools/ffmpeg
[ -s "$out" ] && exit 0
mkdir -p "$(dirname "$out")"
vf="scale='if(gt(iw,ih),-2,384)':'if(gt(iw,ih),384,-2)':flags=bicubic"
[ "$fps" != native ] && vf="$vf,fps=$fps"
lr=""; case "$in" in */opendl/egodex/*) lr="-lowres 1";; esac
part="$out.part.$(hostname -s).$$"   # unique per process: two nodes may race on the same file
if $FF -nostdin -v error -y $lr -threads 2 -i "$in" -an -vf "$vf" -c:v libx264 -preset ultrafast -crf 23 -g $gop -pix_fmt yuv420p \
     -threads 2 -movflags +faststart -f mp4 "$part" 2> "$out.err"; then
  mv "$part" "$out"; rm -f "$out.err"
else rm -f "$part"; echo "$in" >> /data/llama_vjepa_lm/training/scripts/data_gen/prep_failed.txt; fi
