---
layout: post
title:  "Building a \"no sweat\" Word Embeddings"
description: This article introduces a simple word embeddings model trained from nltk brown word corpus.
tags: ml nlp introduction
usemathjax: true
---

# Introduction

The large number of English words can make language-based applications daunting.  To cope with this, it is helpful to have a clustering or embedding of these words, so that words with similar meanings are clustered together, or have embeddings that are close to one another. 

But how can we get at the meanings of words?
  > "You shall know a word by the company it keeps."
  >    - John Firth (1957)

That is, words that tend to appear in similar contexts are likely to be related.  

In this article, I will investigate this idea by coming up with a **100 dimensional embedding of words** that is based on co-occurrence statistics.

The description here assumes you are using Python with NLTK.


```python
import numpy as np
import pandas as pd

import nltk
nltk.download('brown')
from nltk.corpus import brown
from nltk.corpus import stopwords

import collections
from string import digits, punctuation
```

    [nltk_data] Downloading package brown to
    [nltk_data]     /Users/galaxie500/nltk_data...
    [nltk_data]   Package brown is already up-to-date!


- Prepare data

First, download the Brown corpus (using `nltk.corpus`).  This is a collection of text samples from a wide range of sources, with a total of over a million words.  Calling `brown.words()` returns this text in one long list, which is useful.


```python
print(type(brown.words()))
words = list(brown.words())
print(f"Length of raw Brown corpus: {len(words)}")
```

    <class 'nltk.corpus.reader.util.ConcatenatedCorpusView'>
    Length of raw Brown corpus: 1161192



## (a) Step-by-Step: build a 100-dimensional embeddings

- Lowercase and Remove stopwords, punctuation

Remove stopwords and punctuation, make everything lowercase, and count how often each word occurs. Use this to come up with two lists:
  1. *A vocabulary V*, consisting of a few thousand (e.g., 5000) of the most commonly-occurring words.
  2. *A shorter list C* of at most 1000 of the most commonly-occurring words, which we shall call context words.


```python
# clean text data
def preprocessing(corpus):
    for i in range(len(corpus)):
        corpus[i] = corpus[i].translate(str.maketrans('', '', punctuation))
        #corpus[i] = corpus[i].translate(str.maketrans('', '', digits))
        corpus[i] = corpus[i].lower()
    
    stop_words = set(stopwords.words())
    corpus = [w for w in corpus if w!='' if not w in stop_words] #remove empty string and stopwords
    return corpus
```


```python
words = preprocessing(words)
```


- Generate *vocabulary V* and *shorter list C*


```python
def extract_frequent_words(word_list, freq=1000):
    count = collections.Counter(word_list)
    word_count = {'Word':list(count.keys()), 'Count':list(count.values())}
    df = pd.DataFrame(data=word_count)
    df = df.sort_values(by='Count', ascending=False)
    selected = df[0:freq]
    selected = selected.sort_index()
    return selected
```


```python
C_df = extract_frequent_words(words, freq=1000)
C_list = C_df.Word.values
```


```python
V_df = extract_frequent_words(words, freq=5000)
V_df.columns = ['V_words', 'window_Count']
V_list = V_df.V_words.values
```


For each word $$w \in V$$, and each occurrence of it in the text stream, look at the surrounding window of four words (two before, two after):

$$w_1 \quad w_2 \quad w \quad w_3 \quad w_4$$

Keep count of how often context words from $$C$$ appear in these positions around word $$w$$.  That is, for $$w \in V$$, $$c \in C$$, define:

$$n(w,c) = $$ # of times c occurs in a window around $$w$$

Using these counts, construct the probability distribution $$Pr(c \vert w)$$ of context words around $$w$$(for each $$w \in V$$), as well as the overall distribution *Pr(c)* of context words.  These are distributions over $$C$$.


- Calculate $$Pr(c \vert w)$$


```python
# track all the positions that word w showed up in the text stream
V_pos = [[x for x, n in enumerate(words) if n == w] for w in V_list]
```


```python
Windows = []
context_word_count = []
for pos in V_pos:
    window = [] # context words surrounding w for w in V
    for i in pos:
        if i==0:
            cur_window = words[1:3]
        elif i==1:
            cur_window = [words[0]] + words[2:4]
        else:
            cur_window = words[i-2:i] + words[i+1:i+3]
        
        # exclude duplicate context words in a single window
        cur_unique = []
        for j in cur_window:
            if j not in cur_unique:
                cur_unique.append(j)
        
        window += cur_unique
    
    # count occurrence of each context word for words in window w
    context_word_count += list(collections.Counter(window).values())
    
    # remove duplicated context words
    context_word_unique = []
    for k in window:
        if k not in context_word_unique:
            context_word_unique.append(k)
            
    Windows.append(context_word_unique)
```


```python
V_df['context_words'] = Windows
V_df = V_df.explode('context_words')
V_df['countext_word_count'] = context_word_count
V_df['Pr(c|w)'] = V_df['countext_word_count'].astype('float')/V_df['window_Count']
V_df['in_C'] = [1 if i in C_list else 0 for i in V_df.context_words.values]
```


```python
# drop all the rows with context words that are not in our shorter list C
V_df = V_df[V_df.in_C==1]

# drop boolean column 'in_C' after we collect all context words belonged to C
V_df.drop(columns='in_C')
```



Output:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V_words</th>
      <th>window_Count</th>
      <th>context_words</th>
      <th>countext_word_count</th>
      <th>Pr(c|w)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>general</td>
      <td>1</td>
      <td>0.006452</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>none</td>
      <td>1</td>
      <td>0.006452</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>future</td>
      <td>1</td>
      <td>0.006452</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>doctor</td>
      <td>1</td>
      <td>0.006452</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>cent</td>
      <td>1</td>
      <td>0.006452</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>person</td>
      <td>1</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>took</td>
      <td>1</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>energy</td>
      <td>1</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>said</td>
      <td>1</td>
      <td>0.052632</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>interested</td>
      <td>1</td>
      <td>0.052632</td>
    </tr>
  </tbody>
</table>
<p>461410 rows × 5 columns</p>
</div>



- Calculate *Pr(c)*


```python
# calculate distribution of context words in C
C_list_uniq = list(V_df.context_words.unique())
Pr_C = {}
for c in C_list_uniq:
    Pr_C[c] = words.count(c) / len(words)
```


```python
V_df['Pr(c)'] = V_df.loc[:, 'context_words'].apply(lambda x: Pr_C[x])
V_df
```



Output:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V_words</th>
      <th>window_Count</th>
      <th>context_words</th>
      <th>countext_word_count</th>
      <th>Pr(c|w)</th>
      <th>in_C</th>
      <th>Pr(c)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>general</td>
      <td>1</td>
      <td>0.006452</td>
      <td>1</td>
      <td>0.000950</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>none</td>
      <td>1</td>
      <td>0.006452</td>
      <td>1</td>
      <td>0.000206</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>future</td>
      <td>1</td>
      <td>0.006452</td>
      <td>1</td>
      <td>0.000433</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>doctor</td>
      <td>1</td>
      <td>0.006452</td>
      <td>1</td>
      <td>0.000191</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>cent</td>
      <td>1</td>
      <td>0.006452</td>
      <td>1</td>
      <td>0.000296</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>person</td>
      <td>1</td>
      <td>0.052632</td>
      <td>1</td>
      <td>0.000332</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>took</td>
      <td>1</td>
      <td>0.052632</td>
      <td>1</td>
      <td>0.000813</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>energy</td>
      <td>1</td>
      <td>0.052632</td>
      <td>1</td>
      <td>0.000191</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>said</td>
      <td>1</td>
      <td>0.052632</td>
      <td>1</td>
      <td>0.003741</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>interested</td>
      <td>1</td>
      <td>0.052632</td>
      <td>1</td>
      <td>0.000200</td>
    </tr>
  </tbody>
</table>
<p>461410 rows × 7 columns</p>
</div>



- Calculate *pointwise mutual information*

Represent each vocabulary item $$w$$ by a |C|-dimensional vector $$\phi(w)$$, whose $$c$$’th coordinate is:

$$\phi_c(w) = max(0, \frac{logPr(c|w)}{Pr(c)})$$

This is known as the (positive) *pointwise mutual information*, and has been quite successful in work on word embeddings.



```python
import math 

def pointwise_mutual_info(row):
    a = row['Pr(c|w)']
    b = row['Pr(c)']
    res = math.log(a/b)
    return max(0, res)
```


```python
V_df['Phi'] = V_df.apply(pointwise_mutual_info, axis =1)
V_df
```



Output:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V_words</th>
      <th>window_Count</th>
      <th>context_words</th>
      <th>countext_word_count</th>
      <th>Pr(c|w)</th>
      <th>in_C</th>
      <th>Pr(c)</th>
      <th>Phi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>general</td>
      <td>1</td>
      <td>0.006452</td>
      <td>1</td>
      <td>0.000950</td>
      <td>1.915628</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>none</td>
      <td>1</td>
      <td>0.006452</td>
      <td>1</td>
      <td>0.000206</td>
      <td>3.444097</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>future</td>
      <td>1</td>
      <td>0.006452</td>
      <td>1</td>
      <td>0.000433</td>
      <td>2.701278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>doctor</td>
      <td>1</td>
      <td>0.006452</td>
      <td>1</td>
      <td>0.000191</td>
      <td>3.521058</td>
    </tr>
    <tr>
      <th>1</th>
      <td>county</td>
      <td>155</td>
      <td>cent</td>
      <td>1</td>
      <td>0.006452</td>
      <td>1</td>
      <td>0.000296</td>
      <td>3.082803</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>person</td>
      <td>1</td>
      <td>0.052632</td>
      <td>1</td>
      <td>0.000332</td>
      <td>5.066159</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>took</td>
      <td>1</td>
      <td>0.052632</td>
      <td>1</td>
      <td>0.000813</td>
      <td>4.170775</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>energy</td>
      <td>1</td>
      <td>0.052632</td>
      <td>1</td>
      <td>0.000191</td>
      <td>5.620044</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>said</td>
      <td>1</td>
      <td>0.052632</td>
      <td>1</td>
      <td>0.003741</td>
      <td>2.644005</td>
    </tr>
    <tr>
      <th>47101</th>
      <td>letch</td>
      <td>19</td>
      <td>interested</td>
      <td>1</td>
      <td>0.052632</td>
      <td>1</td>
      <td>0.000200</td>
      <td>5.571254</td>
    </tr>
  </tbody>
</table>
<p>461410 rows × 8 columns</p>
</div>



- Dimension Reduction

#### Before conducting dimension reduction, we have to create a sparse matrix with rows as V_words and columns as context words, the value will be the *pointwise mutual information*.


```python
sparse_table = pd.pivot_table(V_df, index = 'V_words', columns = 'context_words', values = 'Phi')
sparse_table
```



Output:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>context_words</th>
      <th>1</th>
      <th>10</th>
      <th>100</th>
      <th>12</th>
      <th>15</th>
      <th>1959</th>
      <th>1960</th>
      <th>1961</th>
      <th>2</th>
      <th>20</th>
      <th>...</th>
      <th>written</th>
      <th>wrong</th>
      <th>wrote</th>
      <th>year</th>
      <th>years</th>
      <th>yes</th>
      <th>yet</th>
      <th>york</th>
      <th>young</th>
      <th>youre</th>
    </tr>
    <tr>
      <th>V_words</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.973217</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.629721</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.965613</td>
      <td>3.781690</td>
      <td>3.340674</td>
      <td>1.984232</td>
      <td>3.333348</td>
      <td>NaN</td>
      <td>3.328978</td>
      <td>3.355489</td>
      <td>4.750349</td>
      <td>2.213074</td>
      <td>...</td>
      <td>1.830082</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.323740</td>
      <td>0.651427</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.156607</td>
      <td>0.913791</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>3.648159</td>
      <td>NaN</td>
      <td>4.766189</td>
      <td>4.731503</td>
      <td>4.471181</td>
      <td>NaN</td>
      <td>3.550520</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.449519</td>
      <td>...</td>
      <td>2.967915</td>
      <td>NaN</td>
      <td>2.80637</td>
      <td>2.614275</td>
      <td>3.804163</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.294440</td>
      <td>2.051624</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>100</th>
      <td>3.340674</td>
      <td>4.766189</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.919708</td>
      <td>3.397186</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.734538</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>4.697981</td>
      <td>NaN</td>
      <td>4.989332</td>
      <td>NaN</td>
      <td>4.694324</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.223713</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>youth</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.502838</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>youve</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>3.928008</td>
      <td>4.105148</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.056206</td>
      <td>NaN</td>
      <td>2.92709</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.947681</td>
    </tr>
    <tr>
      <th>zen</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>zero</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>zg</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 1000 columns</p>
</div>



#### Here, we will be doing *Singular Value Decomposition*(SVD) to decompose the above sparse matrix to 100 dimension word representation.


```python
# fill NaN with 0s
sparse_table = sparse_table.fillna(0)
```


```python
from sklearn.decomposition import TruncatedSVD
#from sklearn.random_projection import sparse_random_matrix

X = np.asarray(sparse_table)

svd = TruncatedSVD(n_components = 100, n_iter = 5, random_state = 42)
svd.fit(X)

X_new = svd.fit_transform(X)
df_new = pd.DataFrame(X_new, index = sparse_table.index)
df_new
```



Output:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>90</th>
      <th>91</th>
      <th>92</th>
      <th>93</th>
      <th>94</th>
      <th>95</th>
      <th>96</th>
      <th>97</th>
      <th>98</th>
      <th>99</th>
    </tr>
    <tr>
      <th>V_words</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.542319</td>
      <td>-1.995680</td>
      <td>1.381643</td>
      <td>-2.909863</td>
      <td>-3.406599</td>
      <td>-1.401489</td>
      <td>-3.144874</td>
      <td>0.091159</td>
      <td>0.404984</td>
      <td>-2.519494</td>
      <td>...</td>
      <td>-0.272699</td>
      <td>0.328477</td>
      <td>0.218683</td>
      <td>-0.656883</td>
      <td>-0.016048</td>
      <td>-1.542390</td>
      <td>1.702155</td>
      <td>-0.510453</td>
      <td>-1.331743</td>
      <td>0.531266</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23.523418</td>
      <td>-12.896316</td>
      <td>11.089762</td>
      <td>-3.408536</td>
      <td>-9.895360</td>
      <td>-3.761695</td>
      <td>0.060779</td>
      <td>7.032133</td>
      <td>-1.337786</td>
      <td>-3.057331</td>
      <td>...</td>
      <td>-1.908982</td>
      <td>1.097818</td>
      <td>-1.488350</td>
      <td>-1.397893</td>
      <td>1.135940</td>
      <td>-0.119767</td>
      <td>0.577449</td>
      <td>-0.072922</td>
      <td>1.941455</td>
      <td>1.521440</td>
    </tr>
    <tr>
      <th>10</th>
      <td>17.485695</td>
      <td>-4.392681</td>
      <td>13.806938</td>
      <td>-0.205896</td>
      <td>-9.351523</td>
      <td>-3.323489</td>
      <td>2.553425</td>
      <td>-0.650727</td>
      <td>4.434021</td>
      <td>-3.258592</td>
      <td>...</td>
      <td>-1.889507</td>
      <td>-1.311489</td>
      <td>0.508911</td>
      <td>-1.792407</td>
      <td>0.841425</td>
      <td>1.114498</td>
      <td>0.939590</td>
      <td>1.554220</td>
      <td>-1.658337</td>
      <td>1.261531</td>
    </tr>
    <tr>
      <th>100</th>
      <td>12.115965</td>
      <td>-3.480162</td>
      <td>7.104189</td>
      <td>-1.246714</td>
      <td>-4.169442</td>
      <td>-0.304919</td>
      <td>1.910621</td>
      <td>-2.503645</td>
      <td>2.942702</td>
      <td>-1.915575</td>
      <td>...</td>
      <td>1.479620</td>
      <td>0.117896</td>
      <td>-0.171019</td>
      <td>1.062107</td>
      <td>1.123733</td>
      <td>-0.680444</td>
      <td>-1.128887</td>
      <td>-1.518986</td>
      <td>1.168567</td>
      <td>0.436447</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>6.748229</td>
      <td>-3.263267</td>
      <td>3.999501</td>
      <td>-1.575692</td>
      <td>-4.276787</td>
      <td>-0.313027</td>
      <td>1.567308</td>
      <td>-1.057089</td>
      <td>2.655190</td>
      <td>-2.324639</td>
      <td>...</td>
      <td>0.150966</td>
      <td>1.584698</td>
      <td>1.008214</td>
      <td>1.805991</td>
      <td>-1.644718</td>
      <td>0.632643</td>
      <td>0.848221</td>
      <td>-0.695705</td>
      <td>0.090873</td>
      <td>-0.922286</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>youth</th>
      <td>13.327412</td>
      <td>0.517403</td>
      <td>-2.878062</td>
      <td>-0.924167</td>
      <td>2.294374</td>
      <td>-0.957506</td>
      <td>-0.091582</td>
      <td>-3.612247</td>
      <td>-1.984099</td>
      <td>-1.501401</td>
      <td>...</td>
      <td>0.115512</td>
      <td>0.774741</td>
      <td>1.203958</td>
      <td>3.243366</td>
      <td>-1.524132</td>
      <td>-0.470964</td>
      <td>-1.921117</td>
      <td>1.201018</td>
      <td>2.303047</td>
      <td>-0.377757</td>
    </tr>
    <tr>
      <th>youve</th>
      <td>13.042776</td>
      <td>8.514331</td>
      <td>-6.500516</td>
      <td>2.708451</td>
      <td>-5.992631</td>
      <td>1.492063</td>
      <td>0.820251</td>
      <td>-0.925629</td>
      <td>0.217164</td>
      <td>-1.467112</td>
      <td>...</td>
      <td>-0.712831</td>
      <td>-1.747105</td>
      <td>-1.294000</td>
      <td>0.663011</td>
      <td>-0.229572</td>
      <td>0.423359</td>
      <td>-1.324096</td>
      <td>-0.035515</td>
      <td>-0.995549</td>
      <td>-0.576969</td>
    </tr>
    <tr>
      <th>zen</th>
      <td>4.458987</td>
      <td>-1.740182</td>
      <td>-1.933253</td>
      <td>-1.843287</td>
      <td>0.210213</td>
      <td>-1.749685</td>
      <td>-0.096430</td>
      <td>-1.115299</td>
      <td>-0.678715</td>
      <td>1.249615</td>
      <td>...</td>
      <td>-1.146979</td>
      <td>-0.144040</td>
      <td>-0.256134</td>
      <td>-0.293627</td>
      <td>0.389218</td>
      <td>0.439939</td>
      <td>-0.344095</td>
      <td>-0.105342</td>
      <td>0.127606</td>
      <td>-1.844329</td>
    </tr>
    <tr>
      <th>zero</th>
      <td>6.126870</td>
      <td>0.258309</td>
      <td>0.348387</td>
      <td>-3.972732</td>
      <td>-0.751711</td>
      <td>-1.251426</td>
      <td>-2.420377</td>
      <td>1.122941</td>
      <td>0.488333</td>
      <td>-0.156903</td>
      <td>...</td>
      <td>-0.210953</td>
      <td>0.934246</td>
      <td>-1.952539</td>
      <td>1.080573</td>
      <td>0.198827</td>
      <td>-0.895675</td>
      <td>-1.193648</td>
      <td>-0.156535</td>
      <td>-0.424499</td>
      <td>1.452921</td>
    </tr>
    <tr>
      <th>zg</th>
      <td>3.466457</td>
      <td>-0.743264</td>
      <td>-0.012059</td>
      <td>-1.887242</td>
      <td>-0.635641</td>
      <td>-1.200829</td>
      <td>-3.495046</td>
      <td>1.720301</td>
      <td>0.282265</td>
      <td>0.022144</td>
      <td>...</td>
      <td>0.220086</td>
      <td>-0.863757</td>
      <td>0.438066</td>
      <td>1.188732</td>
      <td>0.514399</td>
      <td>0.546616</td>
      <td>-0.517487</td>
      <td>0.494538</td>
      <td>0.355671</td>
      <td>-0.530369</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 100 columns</p>
</div>




## (b) Nearest neighbor results

Pick a collection of $$25$$ words $$w \in V$$.  For each $$w$$, return its nearest neighbor $$w′\neq w$$ in $$V$$.  A popular distance measure to use for this is *cosine distance*:

$$1−\frac{\phi(w)\phi(w′)}{\left\Vert\phi(w)\right\Vert\left\Vert\phi(w′)\right\Vert}$$


```python
from sklearn.metrics.pairwise import cosine_similarity

S = 1 - cosine_similarity(X_new, X_new)
S_df = pd.DataFrame(S, index=sparse_table.index, columns=sparse_table.index)

np.fill_diagonal(S_df.values, 1)
S_df
```



Output:
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>V_words</th>
      <th>0</th>
      <th>1</th>
      <th>10</th>
      <th>100</th>
      <th>1000</th>
      <th>10000</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>...</th>
      <th>youll</th>
      <th>young</th>
      <th>younger</th>
      <th>youngsters</th>
      <th>youre</th>
      <th>youth</th>
      <th>youve</th>
      <th>zen</th>
      <th>zero</th>
      <th>zg</th>
    </tr>
    <tr>
      <th>V_words</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>0.496496</td>
      <td>0.612922</td>
      <td>0.704027</td>
      <td>0.699172</td>
      <td>0.709779</td>
      <td>0.758037</td>
      <td>0.676296</td>
      <td>0.783534</td>
      <td>0.703629</td>
      <td>...</td>
      <td>0.742794</td>
      <td>0.753591</td>
      <td>0.892090</td>
      <td>0.733692</td>
      <td>0.792452</td>
      <td>0.849207</td>
      <td>0.785617</td>
      <td>0.843505</td>
      <td>0.567424</td>
      <td>0.702720</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.496496</td>
      <td>1.000000</td>
      <td>0.270032</td>
      <td>0.407476</td>
      <td>0.426005</td>
      <td>0.598368</td>
      <td>0.393145</td>
      <td>0.335564</td>
      <td>0.439774</td>
      <td>0.364654</td>
      <td>...</td>
      <td>0.725060</td>
      <td>0.535398</td>
      <td>0.794009</td>
      <td>0.707147</td>
      <td>0.796263</td>
      <td>0.730820</td>
      <td>0.776692</td>
      <td>0.674079</td>
      <td>0.570495</td>
      <td>0.772490</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.612922</td>
      <td>0.270032</td>
      <td>1.000000</td>
      <td>0.336588</td>
      <td>0.410105</td>
      <td>0.531437</td>
      <td>0.251751</td>
      <td>0.242511</td>
      <td>0.308180</td>
      <td>0.196244</td>
      <td>...</td>
      <td>0.689861</td>
      <td>0.578702</td>
      <td>0.622325</td>
      <td>0.658268</td>
      <td>0.763801</td>
      <td>0.776299</td>
      <td>0.775649</td>
      <td>0.812817</td>
      <td>0.780041</td>
      <td>0.923918</td>
    </tr>
    <tr>
      <th>100</th>
      <td>0.704027</td>
      <td>0.407476</td>
      <td>0.336588</td>
      <td>1.000000</td>
      <td>0.471953</td>
      <td>0.529019</td>
      <td>0.433712</td>
      <td>0.423225</td>
      <td>0.473367</td>
      <td>0.443475</td>
      <td>...</td>
      <td>0.665715</td>
      <td>0.581621</td>
      <td>0.759004</td>
      <td>0.760093</td>
      <td>0.755221</td>
      <td>0.645551</td>
      <td>0.737462</td>
      <td>0.677804</td>
      <td>0.749595</td>
      <td>0.889030</td>
    </tr>
    <tr>
      <th>1000</th>
      <td>0.699172</td>
      <td>0.426005</td>
      <td>0.410105</td>
      <td>0.471953</td>
      <td>1.000000</td>
      <td>0.694203</td>
      <td>0.532895</td>
      <td>0.573881</td>
      <td>0.464923</td>
      <td>0.510910</td>
      <td>...</td>
      <td>0.695655</td>
      <td>0.760756</td>
      <td>0.832903</td>
      <td>0.862806</td>
      <td>0.841124</td>
      <td>0.666632</td>
      <td>0.854410</td>
      <td>0.928525</td>
      <td>0.830016</td>
      <td>0.923236</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>youth</th>
      <td>0.849207</td>
      <td>0.730820</td>
      <td>0.776299</td>
      <td>0.645551</td>
      <td>0.666632</td>
      <td>0.811296</td>
      <td>0.802807</td>
      <td>0.837998</td>
      <td>0.775515</td>
      <td>0.782112</td>
      <td>...</td>
      <td>0.603810</td>
      <td>0.388828</td>
      <td>0.612672</td>
      <td>0.738526</td>
      <td>0.614601</td>
      <td>1.000000</td>
      <td>0.632270</td>
      <td>0.636307</td>
      <td>0.708969</td>
      <td>0.846091</td>
    </tr>
    <tr>
      <th>youve</th>
      <td>0.785617</td>
      <td>0.776692</td>
      <td>0.775649</td>
      <td>0.737462</td>
      <td>0.854410</td>
      <td>0.749709</td>
      <td>0.773328</td>
      <td>0.804094</td>
      <td>0.916708</td>
      <td>0.787269</td>
      <td>...</td>
      <td>0.419042</td>
      <td>0.478772</td>
      <td>0.599636</td>
      <td>0.814881</td>
      <td>0.294050</td>
      <td>0.632270</td>
      <td>1.000000</td>
      <td>0.650591</td>
      <td>0.714497</td>
      <td>0.884929</td>
    </tr>
    <tr>
      <th>zen</th>
      <td>0.843505</td>
      <td>0.674079</td>
      <td>0.812817</td>
      <td>0.677804</td>
      <td>0.928525</td>
      <td>0.815137</td>
      <td>0.739136</td>
      <td>0.813954</td>
      <td>0.856319</td>
      <td>0.872752</td>
      <td>...</td>
      <td>0.774752</td>
      <td>0.655871</td>
      <td>0.794671</td>
      <td>0.668088</td>
      <td>0.783287</td>
      <td>0.636307</td>
      <td>0.650591</td>
      <td>1.000000</td>
      <td>0.780444</td>
      <td>0.809703</td>
    </tr>
    <tr>
      <th>zero</th>
      <td>0.567424</td>
      <td>0.570495</td>
      <td>0.780041</td>
      <td>0.749595</td>
      <td>0.830016</td>
      <td>0.770043</td>
      <td>0.779864</td>
      <td>0.574442</td>
      <td>0.784010</td>
      <td>0.758421</td>
      <td>...</td>
      <td>0.762509</td>
      <td>0.685187</td>
      <td>0.872117</td>
      <td>0.858662</td>
      <td>0.745964</td>
      <td>0.708969</td>
      <td>0.714497</td>
      <td>0.780444</td>
      <td>1.000000</td>
      <td>0.531597</td>
    </tr>
    <tr>
      <th>zg</th>
      <td>0.702720</td>
      <td>0.772490</td>
      <td>0.923918</td>
      <td>0.889030</td>
      <td>0.923236</td>
      <td>0.829799</td>
      <td>0.939233</td>
      <td>0.771256</td>
      <td>0.887662</td>
      <td>0.925314</td>
      <td>...</td>
      <td>0.812688</td>
      <td>0.861458</td>
      <td>0.917434</td>
      <td>0.817756</td>
      <td>0.890121</td>
      <td>0.846091</td>
      <td>0.884929</td>
      <td>0.809703</td>
      <td>0.531597</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 5000 columns</p>
</div>




```python
to_check = ['communism', 'autumn', 'cigarette', 'pulmonary', 'mankind', 
            'africa', 'chicago', 'revolution', 'september', 'chemical', 
            'detergent', 'dictionary', 'storm', 'worship']
```


```python
S_dict = {}
for t in to_check:
    S_dict[t] = S_df[t].idxmin()
    
for d in S_dict:
    print (f'{d} ------> {S_dict[d]}')
```
Output:

    communism ------> almost
    autumn ------> dawn
    cigarette ------> fingers
    pulmonary ------> artery
    mankind ------> nation
    africa ------> western
    chicago ------> club
    revolution ------> movement
    september ------> december
    chemical ------> drugs
    detergent ------> tubes
    dictionary ------> text
    storm ------> wedding
    worship ------> organized


As we can see from above, some of the nearest neighbors do make sense, like `cigarette -> smelled`, `africa -> asia`, `september -> december`,  `dictionary -> text` and `storm -> summer`.
