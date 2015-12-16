# Rappers
Automatic rap lyric generation tool

# Requirements

- invoke
- tensorflow

# Usage
## Neural Network Language Model

- Training

```
inv train model
```

- Testing

```
inv test model
```

## Construct corpus ##

```
# pip install beautifulsoup
python getlyrics.py -v > output.tsv
```
