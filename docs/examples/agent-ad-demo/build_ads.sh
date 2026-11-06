#!/bin/bash
# Usage: build_ads.sh  (run from video/work). Needs ../c1..c4.mp4, ../c5b.mp4 (or c5.mp4), w1..w6.wav, k*.png, vk*.png, end.png, vend.png
set -e
C5=../c5b.mp4; [ -f $C5 ] || C5=../c5.mp4
IN="-i ../c1.mp4 -i ../c2.mp4 -i ../c3.mp4 -i ../c4.mp4 -i $C5"
VO="-i w1.wav -i w2.wav -i w3.wav -i w4.wav -i w5.wav -i w6.wav"
AUD="[a0][a1][a2][a3][a4]concat=n=5:v=0:a=1,apad=whole_dur=30.1[amb];
[11:a]adelay=0|0,volume=1.5[s1];[12:a]adelay=3900|3900,volume=1.5[s2];[13:a]adelay=8300|8300,volume=1.5[s3];[14:a]adelay=13000|13000,volume=1.5[s4];[15:a]adelay=17700|17700,volume=1.5[s5];[16:a]adelay=22600|22600,volume=1.5[s6];
[amb][s1][s2][s3][s4][s5][s6]amix=inputs=7:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11,aformat=sample_rates=44100:channel_layouts=stereo[aout]"
# ---- 16:9 ----
ffmpeg -loglevel error -y $IN \
 $(for k in k1 k2 k3 k4 k5; do echo "-loop 1 -t 4.65 -i $k.png"; done) -loop 1 -t 8 -i end.png $VO \
 -filter_complex "
$(for i in 0 1 2 3 4; do echo "[$i:v]scale=1280:-2,crop=1280:720,setpts=1.15*PTS,fps=24,format=yuva420p[b$i];[$((i+5)):v]format=yuva420p[c$i];[b$i][c$i]overlay=shortest=1,format=yuv420p[v$i];[$i:a]atempo=0.87,volume=0.45[a$i];"; done)
[10:v]zoompan=z='1+0.0006*on':d=192:s=1280x720:fps=24,format=yuv420p[v5];
[v0][v1][v2][v3][v4][v5]concat=n=6:v=1:a=0[vout];
$AUD" -map "[vout]" -map "[aout]" -t 30 -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac -b:a 160k ../../advert_30s_16x9.mp4
# ---- 9:16 (clip centred on a blurred copy of itself) ----
ffmpeg -loglevel error -y $IN \
 $(for k in vk1 vk2 vk3 vk4 vk5; do echo "-loop 1 -t 4.65 -i $k.png"; done) -loop 1 -t 8 -i vend.png $VO \
 -filter_complex "
$(for i in 0 1 2 3 4; do echo "[$i:v]setpts=1.15*PTS,fps=24,split[s$i][t$i];[s$i]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=40,eq=brightness=-0.15[bg$i];[t$i]scale=1080:-2[fg$i];[bg$i][fg$i]overlay=0:(H-h)/2-200,format=yuva420p[b$i];[$((i+5)):v]format=yuva420p[c$i];[b$i][c$i]overlay=shortest=1,format=yuv420p[v$i];[$i:a]atempo=0.87,volume=0.45[a$i];"; done)
[10:v]zoompan=z='1+0.0005*on':d=192:s=1080x1920:fps=24,format=yuv420p[v5];
[v0][v1][v2][v3][v4][v5]concat=n=6:v=1:a=0[vout];
$AUD" -map "[vout]" -map "[aout]" -t 30 -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac -b:a 160k ../../advert_30s_9x16.mp4
echo built
