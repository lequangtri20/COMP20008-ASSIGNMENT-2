import pandas as pd
import nltk
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from itertools import combinations
nltk.download("stopwords")
from nltk.corpus import stopwords
nltk.download("wordnet")
from nltk.corpus import wordnet
#==============================TEXT PREPROCESSING===============================
#try:
#    from nltk.corpus import stopwords
#except:
#    nltk.download("stopwords")


stpwrds = stopwords.words('english')
abt = pd.read_csv("abt.csv", encoding='ISO-8859-1')
buy = pd.read_csv("buy.csv", encoding='ISO-8859-1')
stemmer= PorterStemmer()
lemmatizer= WordNetLemmatizer()

def clean(name):
    name = name.lower()
    name = name.translate(str.maketrans('', '', string.punctuation))
    name = name.strip()
    
    # lemmatize
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(name))   
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))
    name = lemmatize(wordnet_tagged)
    
    # remove stop words
    tokens = name.split()
    name =  [stemmer.stem(i) for i in tokens if i not in stpwrds]  
    name = " ".join(name)
    return name

def lemmatize(wordnet_tagged):
    lemmatized_sentence = [] 
    for word, tag in wordnet_tagged: 
        if tag is None: 
            # if there is no available tag, append the token as is 
            lemmatized_sentence.append(word) 
        else:         
            # use the tag to lemmatize the token 
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag)) 
            
    return " ".join(lemmatized_sentence) 

# Return the word part of speech tag
def pos_tagger(nltk_tag): 
    if nltk_tag.startswith('J'): 
        return wordnet.ADJ 
    elif nltk_tag.startswith('V'): 
        return wordnet.VERB 
    elif nltk_tag.startswith('N'): 
        return wordnet.NOUN 
    elif nltk_tag.startswith('R'): 
        return wordnet.ADV 
    else:           
        return None
#==============================================================================

abt_blocking = {} # storing independent blocks from abt.csv
buy_blocking = {} # storing independent blocks from buy.csv
abt_key = []
abt_id =[]
buy_key =[]
buy_id = []

# extract block keys and their's id in each file
def independent_blocking(single_block, data):
    for name, prod_id in zip(data["name"], data.iloc[:,0]):
        name = clean(name)

        lst = list(combinations(name.split(),2))
        for b in lst:
            sorted(b)
            if b not in single_block:
                single_block[b]=[prod_id]
            elif b in single_block:
                single_block[b].append(prod_id)

# create final id list for each file based on mutual block keys
def unified_blocking(mutual_block, single_block, id_list, key_list):
    for i in mutual_block:
        for j in single_block[i]:
            id_list.append(j)
            key_list.append(i)


independent_blocking(abt_blocking, abt)
independent_blocking(buy_blocking, buy)
mutual_block = [a for a in abt_blocking if a in buy_blocking]
unified_blocking(mutual_block, abt_blocking, abt_id, abt_key)
unified_blocking(mutual_block, buy_blocking, buy_id, buy_key)

pd.DataFrame({"block_key":abt_key, 
              "product_id":abt_id}).to_csv(r'abt_blocks.csv', index = False)
pd.DataFrame({"block_key":buy_key, 
              "product_id":buy_id}).to_csv(r'buy_blocks.csv', index = False)
