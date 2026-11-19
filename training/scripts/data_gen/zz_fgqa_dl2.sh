#!/bin/bash
# (10-19, v2: decoder / filter threads 1 - automatic threading made 25 ffmpegs fight over 96 cores) PLM-FGQA videos: worker W of N over dl/fgqa/list.tsv. Per video: the video-only stream (<= 480p) with yt-dlp's own
# downloader (the static ffmpeg segfaults on network input), then every segment cut locally straight to the training format
# (short side 384, 15 fps, GOP 15, as repo/scripts/prep/transcode_one.sh) -> prep/fgqa_384_15fps/<source>/<segment_id>.mp4;
# the full video is deleted. Status per video -> dl/fgqa/status_W.tsv (ok / fail + reason); videos already in a status file are skipped.
W=$1; N=$2
R=/data/llama_vjepa_lm; L=$R/dl/fgqa/list.tsv; ST=$R/dl/fgqa/status_$W.tsv; TMP=$R/dl/fgqa/tmp/$W; O=$R/prep/fgqa_384_15fps
YT=$R/tools/ytdlp_venv/bin/yt-dlp; FF=$R/tools/ffmpeg
mkdir -p $TMP; touch $ST
done_ids=$(mktemp); cat $R/dl/fgqa/status_*.tsv 2>/dev/null | cut -f2 | sort -u > $done_ids
awk -v W=$W -v N=$N 'NR % N == W' $L | while IFS=$'\t' read src vid nq segs; do
  grep -qxF "$vid" $done_ids && continue
  [ -f $R/dl/fgqa/STOP ] && break
  rm -f $TMP/*
  err=$(timeout 600 $YT -q --no-warnings --no-playlist -f "bv*[height<=480][ext=mp4]/bv*[height<=480]" -o "$TMP/v.%(ext)s" "https://www.youtube.com/watch?v=$vid" 2>&1 | tail -1 | tr '\t\n' '  ' | cut -c1-150)
  f=$(ls $TMP/v.* 2>/dev/null | grep -v '\.part$' | head -1)
  if [ -z "$f" ]; then echo -e "$src\t$vid\tfail\t$err" >> $ST; sleep 2; continue; fi
  mkdir -p $O/$src; ok=0; bad=0
  IFS=',' read -ra S <<< "$segs"
  for s in "${S[@]}"; do IFS=':' read sid t0 t1 <<< "$s"; out=$O/$src/$sid.mp4
    [ -s $out ] && { ok=$((ok+1)); continue; }
    if $FF -nostdin -v error -y -threads 1 -filter_threads 1 -ss $t0 -to $t1 -i "$f" -an -vf "scale='if(gt(iw,ih),-2,384)':'if(gt(iw,ih),384,-2)':flags=bicubic,fps=15" \
         -c:v libx264 -preset veryfast -crf 23 -g 15 -pix_fmt yuv420p -threads 2 -movflags +faststart -f mp4 $out.part 2>/dev/null && [ -s $out.part ]; then
      mv $out.part $out; ok=$((ok+1)); else rm -f $out.part; bad=$((bad+1)); fi
  done
  rm -f $TMP/*; echo -e "$src\t$vid\tok\t$ok/$((ok+bad))" >> $ST
done
rm -f $done_ids
