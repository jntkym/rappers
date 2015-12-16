# Rappers
Automatic rap lyric generation tool

# Requirements

- invoke
- tensorflow
- juman
- kytea

# Usage

## Obtain lyrics ##

```
# pip install beautifulsoup
python getlyrics.py -v > output.tsv
```

## Construct corpus ##

1) Untar lyrics archive, then run the following command to obtain a file *data/clean_corpus.txt*:
```
python preprocess.py -crawl data/lyrics_shonan_s27_raw.tsv
```
2) Feed the cleaned crawled corpus to juman:
```
juman < data/clean_corpus.txt > data/juman_out.txt
```
3) Process the juman output file:
```
python preprocess.py -juman data/juman_out.txt
```

The preprocessing step is finished. You will have two files in the */data* folder:

- *string_corpus.txt* as a corpus file (one sentence per line) 
- *daihyou_vocab.p* file as a vocabulary file (keys correspond to surface forms, values to 代表表記) - this is used to lookup the embeddings during the LSTM training

## Neural Network Language Model

- Training

```
inv train model
```

- Testing

```
inv test model
```
