# -*- coding: utf-8 -*-

u"""歌詞コーパスの取得


"""

import argparse
import requests
import logging
import urllib, urllib2
import re
from BeautifulSoup import BeautifulSoup

verbose = False
logger = None

def init_logger():
    global logger
    logger = logging.getLogger('GetLyrics')
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)


def getSimilarArtist(artist):
  u"""類似アーティストのリストを取得

  Last.fm APIを使用
  """
  params = {"method":"artist.getSimilar", "artist":artist, 'api_key':'1d4a537bc937e81b88719933eed12ea0'}
  r = requests.get('http://ws.audioscrobbler.com/2.0/', params=params)
  soup = BeautifulSoup(r.content)
  artist_list = soup("name")
  p = re.compile(r"<[^>]*?>")
  
  for i, v in enumerate(artist_list):
    artist_list[i] = p.sub("", str(v))

  if verbose:
    logger.info("Retrieved " + str(len(artist_list)) \
                + " similar artists")
  return artist_list


def getAllLyricsUrl(artistsList):
  u"""アーティストごとのすべての歌詞ページのUrlを取得

  j-lyrics.net
  """
  params = {"kt":"0", "ct":"0", "ka":artist, "ca":"0", "kl":"0", "cl":"0"}
  url = 'http://search.j-lyric.net/?'


def print_artist_list(artist_list):
  u"""Last.fmから取得したアーティストのリストを取得
  """
  for artist_name in artist_list:
    print(artist_name)


def main(args):
  global verbose
  verbose = args.verbose

  artist_list = getSimilarArtist(args.artist)
  artist_list = [args.artist,] + artist_list



if __name__ == '__main__':
  init_logger()
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--artist", default="湘南乃風", help='artist name')
  parser.add_argument('-v', '--verbose', action='store_true', default=False,
                      help='verbose output')
  args = parser.parse_args()
  main(args)

  
