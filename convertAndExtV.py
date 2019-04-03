# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from pydub import AudioSegment
from pydub.playback import play
import youtube_dl

# Download mp3 file from Youtube
a = input("enter url: ")
ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '320', }],
}

with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([a])
    result = ydl.extract_info("{}".format([a]))
