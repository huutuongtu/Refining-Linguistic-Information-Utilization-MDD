import random
import pandas as pd
import ast
import torch

random.seed(1234)

vowels = ['<eps>', 'err', 'ae*', 'uh*', 'ah*', 'aw*', 'ow*', 'eh*', 'er*', 'uw*', 'aa*', 'ao*', 'ey*', 'iy', 'aa', 'ae', 'eh', 'ah', 'ao', 'ih', 'ey', 'aw', 'ay', 'er', 'uw', 'uh', 'oy', 'ow']
consonants = ['<eps>', 'err', 'r*', 'v*' ,'hh*' , 'b*', 'k*', 'zh*', 'w*', 'l*', 'p*', 'g*', 't*', 'dh*', 'z*', 'jh*', 'd*', 'y*', 'n*', 'w', 'dh', 'y', 'hh', 'ch', 'jh', 'th', 'zh', 'd', 'ng', 'b', 'g', 'f', 'k', 'm', 'l', 'n', 's', 'r', 't', 'v', 'z', 'p', 'sh']


#Augment using VC augment in A full text.... but result worse
def linguistic_augment(canonical, mutation_prob=0.15):
    res = ''
    phone_canonical = canonical.split(" ")
    mutation_phone = ''
    for phone in phone_canonical:
        mutation_phone = phone
        if random.random() < mutation_prob:
            if phone in vowels:
                mutation_phone = random.choice(vowels)
            elif phone in consonants:
                mutation_phone = random.choice(consonants)
        res = res + mutation_phone + " "
    return res.strip()


def linguistic_expand_augment(canonical_time, mutation_prob=0.15):
    res = []
    for time_phone in canonical_time:
        time = list(time_phone.keys())[0]
        phone = list(time_phone.values())[0]
        mutation_phone = phone
        if random.random() < mutation_prob:
            if phone in vowels:
                mutation_phone = random.choice(vowels)
            elif phone in consonants:
                mutation_phone = random.choice(consonants)
        res.append({time:mutation_phone})
    return res

"""
cnt = 0
cnt_prob = 0
data = pd.read_csv("train_time.csv")
for i in range(len(data)):
    linguistic = data['Canonical'][i]
    linguistic_augmented = linguistic_augment(linguistic)
    linguistic_expand = ast.literal_eval(data['Canonical_time'][i])
    linguistic_expand_augmented = linguistic_expand_augment(linguistic_expand)
    for each in range(len(linguistic_expand)):
        cnt+=1
        if (linguistic_expand[each]==linguistic_expand_augmented[each]):
            cnt_prob+=1
print(cnt_prob/cnt)
"""