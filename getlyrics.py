# -*- coding: utf-8 -*-

import requests
import urllib, urllib2
import re
from BeautifulSoup import BeautifulSoup

def getSimilarArtist(artist):
  # last.fm api
  # 類似アーティストのリストを取得
  params = {"method":"artist.getSimilar", "artist":artist, 'api_key':'1d4a537bc937e81b88719933eed12ea0'}
  r = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
  soup = BeautifulSoup(r.content)
  artists_list = soup("name")
  p = re.compile(r"<[^>]*?>")
  
  for i, v in enumerate(artists_list):
    artists_list[i] = p.sub("", str(v))

  return artists_lists

def getAllLyricsUrl(artistsList):
  # j-lyrics.net
  # アーティストごとのすべての歌詞ページのUrlを取得
  params = {"kt":"0", "ct":"0", "ka":artist, "ca":"0", "kl":"0", "cl":"0"}
  url = 'http://search.j-lyric.net/?'

if __name__ == '__main__':
  shonan_list = getSimilarArtist("湘南乃風")
  
