#!/usr/bin/env python

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import sentiwordnet as swn
import sys

def main():
        input_text = sys.stdin.read()
        punkt = nltk.data.load('tokenizers/punkt/english.pickle')
        l = WordNetLemmatizer()

        sentences = punkt.tokenize(input_text)
        for sent in sentences:
            tokens = nltk.word_tokenize(sent)
            lemmas = map(l.lemmatize, tokens)
            sum_score = 0
            print("Sentence ---")
            for x in lemmas:
                try:
                    score = list(swn.senti_synsets(x))[0].obj_score()
                except:
                    score = 0
                sum_score += score
                print(x + " : " + str(score))
            print("Total score: " + str(sum_score))

if __name__ == "__main__":
    main()
