---
layout: default
title:  "How I won my first Kaggle Competition"
date:   2022-08-24 00:12:17 +1000
categories: Kaggle, Data Science Competitions, NLP
description: Open Source Data Science PhD
---

Back in 2020 I took part in an online data science competition where the aim was to correctly identify Tweets announcing real disasters from all the other noise on Twitter. This article outlines how I came up with a submission that landed me a spot in the top 5. 

# The Competition

Kaggle is the #1 data science competition website worldwide. It is now owned by Google, who wasted no time promiting/getting useful data for their own products by setting up a "Getting Started" NLP[*](#Footnotes) competition centred around their new [AutoML][automl] product. According to the website, AutoML allows you to _"Train high-quality custom machine learning models with minimal effort and machine learning expertise"_. Basically this platform automates the feature engineering, model selection/training/validation and hyperparameter tuning steps of a normal ML pipeline - all you need to do as a user is supply the data (presuming you happen to have a perfectly cleaned dataset and a task more generic than other peoples' baby pics). What I'm trying to say with that is already much more neatly explained in a [series](https://www.fast.ai/2018/07/23/auto-ml-1/) - [of](https://www.fast.ai/2018/07/23/auto-ml-2/) - [articles](https://www.fast.ai/2018/07/23/auto-ml-3/) by Rachel Thomas on fast.ai, so I'll just quote her here:

>Dealing with data formatting, inconsistencies, and errors is often a messy and tedious process. People will sometimes describe machine learning as separate from data science, as though for machine learning, you can just begin with your nicely cleaned, formatted data set. However, in my experience, the process of cleaning a data set and training a model are usually interwoven: I frequently find issues in the model training that cause me to go back and change the pre-processing for the input data.

Suffice to say that even though this competition required final submissions to be made using AutoML, the bulk of the work still revolved around data cleansing and feature engineering.

# My Submission

I'll start with the tl;dr: you can view my final submission in full [here on kaggle][submission]. Here's the step by step:

1. Correct mislabelled training data
2. Gather, label and append additional training data
3. Use fastai to build a text classifier model
4. Transform tweets using spelling and regex functions
5. Set up and submit trainig data to AutoML

I'll now go over each of the above in detail.

## 1. Correct mislabelled training data
This step is pretty staightforward, but can be very time consuming if done manually (especially if the training dataset is big). Luckily Kaggle has an awesome community, and this is a perfect example of why it pays off to be active on the competition forums. If there's issues with the training data, someone will nearly always discover this eventually and make a post about it. This was the case here as well. In the end the data does not need to be 100% accurately labelled, but depending on if there is any bias in the mislabelling, even chaning a small amount of incorrect labels can make a big difference. The tradeoff, of course, is time.

## 2. Gather, label and append additional training data
I actually came back to this later, once I had a working model and a baseline accuracy score, and it made a tremendous difference to my final results.

The dataset consisted of around 10,000 tweets gathered by keyword search on words such as "quarantine", "ablaze", etc. To gather additional data, I simply repeated this exercise using [Twitter's advanced search][twitter] to collect and label about 300 more tweets. The incidence rate (true positives) of actual disasters as a proportion of returned tweets is quite low. This is likely the main reason why adding more training data (even if it's only ~5%) made a big difference to model accuracy.

Additionally, as sumbissions were limited to using Google's automated neural net search to find the best model for the given dataset, adding more training data was one of the best ways to get ahead of the competition.

## 3. Use fastai to build a text classifier model
Fastai, as the name suggests, is a fantastic library for quickly getting a working ML model for many tasks when working with text or image data. Not only is it fast, but often the default parameters actually yield close to state-of-the art results.

Even though I wouldn't be able to use fastai or any other library (pytorch, keras, etc.) to build a model for my final submission, AutoML is actually quite terrible for iterative model development. For example, if I wanted to add a new text cleaning function to see if it improves my model using AutoML, I would have to submit the new dataset to Google's server and wait for a new model to be trained. This can take several hours, terrible for any ML workflow!

With fastai, I was able to quickly get a text classification model set up using the [text learner api](https://docs.fast.ai/text.learner.html). Now instead of waiting hours to see the results of any minor dataset tweaks, I can get an updated baseline in seconds. Each training cycle using fastai on this dataset took about 10 seconds, and even after 3-4 cycles the model was usually not improving anymore. Here's how simple it is to set up:

```python
from fastai.text import *
import pandas as pd

df_train = pd.read_csv('train.csv')

data_clas = (TextList.from_df(df_train, cols='text', vocab=data_lm.vocab)
           #Inputs: the train.csv file
            .split_by_rand_pct(0.1)
           #We randomly split and keep 10% (10,000 reviews) for validation
            .label_from_df(cols='target')          
           #We want to do a language model so we label accordingly
            .databunch(bs=bs))

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('fine_tuned_enc')

learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))
```

All the code is hosted in [this kaggle notebook](https://www.kaggle.com/code/chardo/disaster-tweets-fast-ai). After a few more cycles of freezing and fitting, the model achieved an accuracy score of 0.816 (i.e. 81.6% correct labels), which isn't far off from our final model's score (which incorporates all the additional training data and text cleaning) of 0.875!

## 4. Transform tweets using spelling and regex functions
Besides gathering more tweets, data transformation was one of the only ways to improve performance. The field of NLP has been getting a lot of attention recently, and a lot of great resources are available on the topic. Great topics to explore further include text stemming, lemmatization, spelling and regex. In most scenarios the current best practicies will generally yield the best results. For my final submission, I used the methods defined in the [NLTK 3 Cookbook][nltk3], a great resource for text transformation and cleaning. Here's how to set this up for any generic text dataset:

```python
import nltk
from nltk import word_tokenize
import enchant
from tqdm import tqdm
from nltk.metrics import edit_distance

replacement_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would'),
]

class RegexpReplacer(object):
    # Replaces regular expression in a text.
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]
    
    def replace(self, text):
        s = text
        
        for (pattern, repl) in self.patterns:
            s = re.sub(pattern, repl, s)
        
        return s

class SpellingReplacer(object):
    """ Replaces misspelled words with a likely suggestion based on shortest
    edit distance
    """
    def __init__(self, dict_name='en', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = max_dist
    
    def replace(self, word):
        if self.spell_dict.check(word):
            return word
        
        suggestions = self.spell_dict.suggest(word)
        
   

     if suggestions and edit_distance(word, suggestions[0]) <= self.max_dist:
            return suggestions[0]
        else:
            return word

def clean_tweet(text) :
    # remove urls
    #text = df.apply(lambda x: re.sub(r'http\S+', '', x))
    text = re.sub(r'http\S+', '', text)

    # replace contractions
    replacer = RegexpReplacer()
    text = replacer.replace(text)

    #split words on - and \
    text = re.sub(r'\b', ' ', text)
    text = re.sub(r'-', ' ', text)

    # replace negations with antonyms

    #nltk.download('punkt')
    tokenizer = nltk.RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(text)

    # spelling correction
    replacer = SpellingReplacer()
    tokens = [replacer.replace(t) for t in tokens]

    # lemmatize/stemming
    wnl = nltk.WordNetLemmatizer()
    tokens = [wnl.lemmatize(t) for t in tokens]
    porter = nltk.PorterStemmer()
    tokens = [porter.stem(t) for t in tokens]
    # filter insignificant words (using fastai)
    # swap word phrases

    text = ' '.join(tokens)
    return(text)
```
Then to apply this to the entire dataset is super straightforward:
```python
tweets = df_train['text']
tqdm.pandas(desc="Cleaning tweets")
tweets_cleaned = tweets.progress_apply(clean_tweet)
df_train['text_clean'] = tweets_cleaned
```
Cleaning tweets: 100%|██████████| 10860/10860 [16:53<00:00, 10.71it/s]

Here's a couple of examples of what how this changes our input data:
>Just happened a terrible car crash > just happen a terribl car crash

>Heard about #earthquake is different cities, stay safe > heard about earthquak is differ citi stay safe

Languague models don't really understand the subtle differences in word endings, tenses and so on. What text cleaning really does is standardize words to make them easier for a machine to interpret. So for example the words "happened", "happens", "happen" are all transformed to "happen".

The final step is to convert our cleaned text data into tokens. There's lots of literature on tokenizing text data that explains this better than I can, but basically this is transforming each work into a unique numeric token that can be read and understood by our neural net. Hence the importance of our text cleaning earlier, else the same word would be tokenized multiple times for each ending/spelling (happened, happens, happen). For this I used the spacy library:

```python
import spacy
train = []
tokenizer = Tokenizer()
tok = SpacyTokenizer('en')
for line in tqdm(df_train.text_clean):
    lne = ' '.join(tokenizer.process_text(line, tok))
    train.append(lne)

df_train['text_tokens'] = train
```

100%|██████████| 10860/10860 [00:02<00:00, 4245.57it/s]

## 5. Set up and submit trainig data to AutoML

Finally we're ready to train our model! This was my first time using Goolge Cloud and AutoML, but luckily as part of the competition a super handy [AutoML Getting Started Notebook](https://www.kaggle.com/code/yufengg/automl-getting-started-notebook) was provided. I copied this workflow and followed the additional instructions to train my model on AutoML. Unlike on most ML projects, there really isn't much to discuss about architecture here, as AutoML is mainly a black box. I'll just list the main steps in the AutoML workflow here:

1. Upload data to a Google Cloud Blob
2. Create a model instance using `AutoMLWrapper`
3. Import our uploaded dataset
4. Train the AutoML model
5. Use the model api for making predictions

And that's it!

# Summary
So that's an overview of my first winning Kaggle sumbission. We went over the basic steps of setting up an iterative model development workflow using fastai, NLP best practices for cleaning and toklenizing text data, and automating it all in Google Cloud. In the end my final submission had around 85% accuracy in classifying disaster tweets. However if the AutoML limitation was removed, the fastai model actually achieved better results at around 87.5% accuracy. Really this shows the limitations of a "fully automated" solution such as AutoML - it still requires all the tedious work of collecting and cleaning data and still performes slightly worse than even a basic fastai model, wich can be set up with ~10 lines of code. Still some way to go with fully automating ML workflows...

### <a name="Footnotes"></a>Footnotes
*NLP = Natural Language Processing; the field of machine learning dedicated to interpreting human speech and writing.

[submission]: https://www.kaggle.com/chardo/top-5-winning-automl-submission
[competition]: https://www.kaggle.com/c/nlp-getting-started
[fastai]: https://github.com/fastai/fastai
[automl]: https://cloud.google.com/automl
[nltk3]: https://github.com/japerk/nltk3-cookbook
[twitter]: https://twitter.com/search-advanced?lang=en