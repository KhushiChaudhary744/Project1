#import libraries
import io
import random
import string 
import warnings
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) #downloading packages

#Reading in corpus
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#Tokenisation
sent_tokens = nltk.sent_tokenize(raw) # converts to list 
word_tokens = nltk.word_tokenize(raw)# converts to list

# Preprocessing
lemeeter = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemeeter.lemmatize(token) for token in tokens]

remove_excess = dict((ord(punct), None) for punct in string.punctuation) #removes extras

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_excess)))

# Keyword that Match
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey","Yo",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me","Nice to meet ypu"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response!!!"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

#Response given by Boto
def response(user_response):
    robot_respose=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robot_respose=robot_respose+"I am sorry! I don't understand you!!!"
        return robot_respose
    else:
        robot_respose = robot_respose+sent_tokens[idx]
        return robot_respose


flag=True
print("Boto: My name is Boto!! I will answer your queries about Chatbots. If you want to exit, type Bye!!!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Boto: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("Boto: "+greeting(user_response))
            else:
                print("Boto: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Boto: Bye! take care..")  