import pandas as pd
import string
import nltk
from fuzzywuzzy import fuzz
from nltk.stem import PorterStemmer
nltk.download("stopwords")
from nltk.corpus import stopwords
#try:
#    from nltk.corpus import stopwords
#except ImportError:
#    nltk.download("stopwords")
    
stpwrds = stopwords.words('english')
stemmer = PorterStemmer()

id_abt_list = [] # store all matched idAbt 
id_buy_list = [] # store all matched idBuy
    
abt = pd.read_csv("abt_small.csv", encoding='ISO-8859-1')
buy = pd.read_csv("buy_small.csv", encoding='ISO-8859-1')

# Below is the function used for text preprocessing.
def clean(name):
    # Change all words to lower case.
    name = name.lower()
    
    # Remove all punctuation and strip the space characters.
    name = name.translate(str.maketrans('', '', string.punctuation))
    name = name.strip()

    #Stemming and removing stopwords
    tokens = name.split()
    name =  " ".join([stemmer.stem(i) for i in tokens if i not in stpwrds])
    return name

# Check if 2 product names have the same product codes.
def exact_code(index):
    ida = id_abt_list[index]
    idb = id_buy_list[index]
    namea = clean(list(abt["name"])[list(abt["idABT"]).index(ida)].split()[-1])
    nameb = clean(list(buy["name"])[list(buy["idBuy"]).index(idb)].split()[-1])
    if namea == nameb:
        return True
    
    return False

#==============================================================================

# Main double loop, it will compare the description from abt file to the product
# name in buy file.
for id_abt, name_abt, des_abt in zip(abt['idABT'], abt['name'],\
                                     abt['description']):
    name_abt = clean(name_abt)
    des_abt = clean(des_abt)
    ad_bn_best = -1 # Used to track the best match so far
    best_match_idb = -1 # Used to store the best match idBuy
    
    for id_buy, name_buy, manu, des_buy in zip(buy['idBuy'], buy['name'],\
                                    buy['manufacturer'], buy["description"]):
        name_buy = clean(name_buy)
        
        # Some product descriptions in buy file are empty which cannot be pre-
        # processed.
        if type(des_buy) != float:
            des_buy = clean(des_buy)
            
        manu = clean(manu)   
             
        # End the algorithm early if two products have different manufacturers
        if name_abt.split()[0] not in manu: #type(manu) != float and (
            if (manu in name_abt) or (name_buy.split()[0] in name_abt) or\
            (name_abt.split()[0] in name_buy): 
                pass
            else:
                continue
            

        # If product code in name abt (extracted up until second last letter)
        # appears in name of buy, it is very likey that two products are the 
        # same product, so algorithm can end earlier.
        if name_abt.split()[-1][:len(name_abt.split()[-1])-1] in\
            name_buy.replace(" ", "") or (type(des_buy) != float and\
            name_abt.split()[-1][:len(name_abt.split()[-1])-1] in\
                des_buy.replace(" ", "")):

            if id_buy in id_buy_list:
                index = id_buy_list.index(id_buy)
                
                # In case two existing products are already matched by exact
                # same code, this whole process is ignored, the loop continues.
                if exact_code(index):
                    break
                
                # If they are not, update the match
                id_abt_list[index] = id_abt
                break
            
            id_buy_list.append(id_buy)
            id_abt_list.append(id_abt)
            break
        
        # Direct string comparison is the last filter so that not every possible
        # pairs are compared. This saves a lot executing time. String comparison
        # is taken place between abt's description and buy's name. Best match
        # is also being tracked
        ad_bn = fuzz.token_set_ratio(des_abt, name_buy)
        if ad_bn >= 72:
            if ad_bn > ad_bn_best: 
                ad_bn_best = ad_bn
                best_match_idb = id_buy
    
    # Record the best match pair
    if ad_bn_best >= 72 and id_abt not in id_abt_list and best_match_idb\
        not in id_buy_list:
        id_buy_list.append(best_match_idb)
        id_abt_list.append(id_abt)


# Producing task1a.csv
df_task1a = pd.DataFrame({"idAbt":id_abt_list, "idBuy":id_buy_list})
df_task1a.to_csv("task1a.csv", index=False)

