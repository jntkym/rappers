# Rappers
Automatic rap lyric generation tool

# Requirements

- invoke
- tensorflow
- juman
- kytea
- chainer

# Usage

## Obtain lyrics ##

```
# pip install beautifulsoup
python getlyrics.py -v > output.tsv
```

## Construct corpus ##

1) Extract lyrics archive, then run the following command to obtain a file `data/juman_input.txt`:
```
python preprocess.py -crawl data/lyrics_shonan_s27_raw.tsv
```
2) Feed the cleaned crawled corpus to juman:
```
juman < data/juman_input.txt > data/juman_out.txt
```
3) Process the juman output file:
```
python preprocess.py -juman data/juman_out.txt
```

The preprocessing step is finished. You will have three files in the `/data` folder:

- `string_corpus.txt` as a string corpus file for LSTM training (one sentence per line), each song is separated from the previous one by one line
- `hiragana_corpus.txt` as a hiragana corpus file for FFNN training (one sentence per line), each song is separated from the previous one by one line 
- `daihyou_vocab.p` file as a vocabulary file (keys correspond to surface forms, values to 代表表記) - this is used to lookup the embeddings during the LSTM training

## Neural Network Language Model

- Training

```
inv train model
```

- Testing

```
inv test model
```


## Chainer LSTM LM 
- Training

run the command below at the directory `chainer_model`
 
```
python train_lstm_lm.py (--gpu 0)
```
You should use gpu to train (this code is very slow on cpu)

- Generating lines

```
python generate_seq.py --model trained_model -O output_file N 10000
```



## Rhyme
Make term-rhyme table using `data/string_corpus.txt` and `data/hiragana_corpus.txt`
```
python features/make_term_vowel_table.py -v --unknown-terms <path-to-unknown-terms:optional> > <path-to-output-table>
```

- `data/term_vowel_table.csv`: term to vowel table (each row has `term,vowels`)
- `data/unknown_terms.txt`: terms that did not have hiragana form in `data/hiragana_corpus.txt`. Currently they are filtered out from the table above


## Next line prediction
```
python NextLine.py -f data/sample_nextline_prediction_candidates.txt 
```
After the processing, you will have the result `test_lyrics.txt`.

Note: You may need to comment out the lines below in `NextLine.py`
```
if __name__ == "__main__":
    ...
            temp.pop(0)
            temp.pop(-1)
```

# API Interface
```
http://lotus.kuee.kyoto-u.ac.jp/~otani/hacku15/api/get_lyric.py
```

You can feed a seed by `seed` parameter:

```
http://lotus.kuee.kyoto-u.ac.jp/~otani/hacku15/api/get_lyric.py?s=パスタ
```
