# -*- coding: utf-8 -*-

u"""歌詞コーパスの取得


"""

import argparse
import requests
import logging
import urllib, urllib2
import re
import time
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
  params = {"method":"artist.getSimilar", "artist":artist,
            "api_key":"1d4a537bc937e81b88719933eed12ea0"}
  r = requests.get("http://ws.audioscrobbler.com/2.0/", params=params)
  soup = BeautifulSoup(r.content)
  artist_list = soup("name")
  p = re.compile(r"<[^>]*?>")
  
  for i, v in enumerate(artist_list):
    artist_list[i] = p.sub("", str(v))

  if verbose:
    logger.info("Retrieved " + str(len(artist_list)) \
                + " similar artists")
  return artist_list


def getArtistId(artist):
  u"""j-lyrics.netでのアーティストのIDを取得
  """
  params = {"ka": artist,}
  baseurl = "http://search.j-lyric.net/index.php"
  r = requests.get(baseurl, params=params)
  soup = BeautifulSoup(r.content)

  urls = soup.find("div", id="lyricList").findAll("a")
  r = re.compile(r'http://j-lyric.net/artist/\w+/')

  for url in urls:
    href = url.get("href")
    if href.startswith("http://j-lyric.net/artist/"):
      return href.split("/")[-2]
  if verbose:
    logger.warning(artist + ": Not found")

  return None


def getLyricUrlList(artist):
  u"""アーティストのすべての歌詞ページのUrlを取得

  j-lyrics.net
  """
  artist_id = getArtistId(artist)
  baseurl = "http://j-lyric.net/artist/" + artist_id + "/"
  r = requests.get(baseurl)
  soup = BeautifulSoup(r.content)
  a_tags = soup.find("div", id="lyricList").findAll("a")
  urls = map(lambda t: (t.string, t.get("href")), a_tags)

  # 歌詞以外のリンクを除く
  urls = filter(lambda t: t[1].startswith('/artist/'), urls)
  urls = map(lambda url: (url[0], "http://j-lyric.net" + url[1]),
             urls)
  return urls


def getLyricText(url):
  u"""歌詞を取得して返す

  """
  r = requests.get(url)
  soup = BeautifulSoup(r.content)
  # TODO: refactoring
  text = str(soup.find("p", id="lyricBody"))
  text = text.replace('<p id="lyricBody">', '').replace('</p>', '')
  text = text.replace('\r', '').replace('\n', '')
  return text.replace('\n', '').replace('<br />', '<BR>')


def printArtistList(artist_list):
  u"""Last.fmから取得したアーティストのリストを取得
  """
  for artist_name in artist_list:
    print(artist_name)


def main(args):
  global verbose
  verbose = args.verbose

  artist_list = getSimilarArtist(args.artist)
  artist_list = [args.artist,] + artist_list

  print("artist\ttitle\ttext")
  for artist in artist_list[:args.n_artists]:
    urls = getLyricUrlList(artist)
    if verbose:
      logger.info('{}: {} songs'.format(artist, len(urls)))
    for i, url in enumerate(urls, start=1):
      if verbose:
        if i%10 == 0: logger.info("Wrote " + str(i) + " songs")
      lyric = getLyricText(url[1])
      print("{artist}\t{title}\t{text}".format(
        artist=artist,
        title=url[0].encode("utf-8"),
        text=lyric))
      time.sleep(1.0)  # Wait one second


if __name__ == '__main__':
  init_logger()
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--artist", default="湘南乃風", help="artist name")
  parser.add_argument("-n", "--n-artists", dest="n_artists",
                      default=10,
                      help="max number of artists")
  parser.add_argument("-v", "--verbose", action="store_true", default=False,
                      help="verbose output")
  args = parser.parse_args()
  main(args)

  
