#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("Running")
import pandas as pd
import numpy as np
from glob import glob
import re
import json
from tqdm import tqdm
import gc
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE

import torch



# In[2]:


directories = []
for i in glob("data/train/*/"):
    for j in glob(i+'*/'):
        directories.append(j)
for i in glob("data/test/*/"):
    for j in glob(i+'*/'):
        directories.append(j)


# In[3]:


data = []
for i in directories:
    try:
        with open(i+'data.json', encoding='utf-8') as f:
            data.append(json.load(f))
    except FileNotFoundError:
        continue
labels = []
for i in directories:
    try:
        with open(i+'labels.json', encoding='utf-8') as f:
            labels.append(json.load(f))
    except FileNotFoundError:
        continue


# In[4]:


# 
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-multi-conversational-hate-sentence-transformer')
# In[ ]:


def abc_ST(d,l):
    embeddings = []
    e = model.encode(d['tweet'])
    embeddings.append({
        'tweet_id':d['tweet_id'],
        'embedding':e,
        'label':l[d['tweet_id']]
    })

    for i in d['comments']:
            c = model.encode(i['tweet'])
            embeddings.append({
                'tweet_id':i['tweet_id'],
                'embedding':0.1*e + 0.1* c,
                'label':l[i['tweet_id']]
            })
            if 'replies' in i.keys():
                for j in i['replies']:
                    r = model.encode(j['tweet'])
                    embeddings.append({
                        'tweet_id':j['tweet_id'],
                        'embedding':0.1*e + 0.1* c + 0.3*r, 
                        'label':l[j['tweet_id']]
                    })
    return embeddings


# In[ ]:


data_label = []
#for train
for i in tqdm(range(len(labels))):
    for j in abc_ST(data[i], labels[i]):
        data_label.append(j)

df = pd.DataFrame(data_label, columns = data_label[0].keys(), index = None)
# In[ ]:


X = [np.array(j) for j in df.embedding]
X = np.array(X)
y = df.label


# In[ ]:


kf = KFold(n_splits=10, shuffle = True)

accs = []
f1_macros = []
misclassifieds = []
for i in kf.split(X,y):
    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(X[i[0]], y[i[0]])
    y_pred = classifier.predict(X[i[1]])
    accs.append(float(accuracy_score(y[i[1]], y_pred)))
    f1_macros.append(float(f1_score(y[i[1]], y_pred, average='macro')))
 
d = {
    'Mean Accuracy':np.mean(np.array(accs)).item(),
    'Mean F1_macro':np.mean(np.array(f1_macros)).item(),
}

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X[:5740], y[:5740])
y_pred = classifier.predict(X[5740:])
d['Test Accuracy'] = float(accuracy_score(y[5740:], y_pred))
d['Test Macro F1'] = float(f1_score(y[5740:], y_pred, average='macro'))

d = json.dumps(d, indent=4)

print(d)

pos = []
neg = []
for i in range(len(y)):
    if(y[i]=='HOF'):
        pos.append(X[i])
    else:
        neg.append(X[i])

pos_dist = 0
for i in range(len(pos)):
    for j in range(i,len(pos)):
        pos_dist += np.sqrt(np.sum(np.square(pos[i]-pos[j])))
pos_dist/=(len(pos)*(len(pos)+1)/2)

neg_dist = 0
for i in range(len(neg)):
    for j in range(i,len(neg)):
        neg_dist += np.sqrt(np.sum(np.square(neg[i]-neg[j])))
neg_dist/=(len(neg)*(len(neg)+1)/2)

pos_neg_dist = 0
for i in range(len(pos)):
    for j in range(len(neg)):
        pos_neg_dist += np.sqrt(np.sum(np.square(pos[i]-neg[j])))
pos_neg_dist/=len(pos)*len(neg)

print(f"Positive distance : {pos_dist}")
print(f"Negative distance : {neg_dist}")
print(f"Positive-Negative distance : {pos_neg_dist}")

# feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
# df = pd.DataFrame(X,columns=feat_cols)
# df['y'] = y
# df['label'] = df['y'].apply(lambda i: str(i))
# np.random.seed(42)
# rndperm = np.random.permutation(df.shape[0])

# tsne2d = TSNE(2)
# tsne2d_results = tsne2d.fit_transform(X)
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x=tsne2d_results[:,0], y=tsne2d_results[:,1],
#     hue="y",
#     palette=sns.color_palette("hls", 2),
#     data=df.loc[rndperm,:],
#     legend="full",
#     alpha=0.3
# )
# plt.savefig('outputs/tsnse 2d.png')
# plt.clf()

# feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
# df = pd.DataFrame(X,columns=feat_cols)
# colors  = []
# for i in y[:5740]:
#     if(i == 'HOF'):
#         colors.append('red')
#     else:
#         colors.append('green')

# for i in y[5740:]:
#     if(i == 'HOF'):
#         colors.append('blue')
#     else:
#         colors.append('yellow')
# df['y'] = colors
# df['label'] = y.apply(lambda i: str(i))
# np.random.seed(42)
# rndperm = np.random.permutation(df.shape[0])

# tsne3d = TSNE(3)
# tsne3d_results = tsne3d.fit_transform(X)
# plt.figure(figsize=(16,10))

# x=tsne3d_results[:,0]
# y=tsne3d_results[:,1]
# z = tsne3d_results[:,2]


# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(
#     xs=x, 
#     ys=y, 
#     zs=z, 
#     c=df.loc[rndperm,:]["y"], 
#     cmap='tab10'
# )
# ax.set_xlabel('tsne-one')
# ax.set_ylabel('tsne-two')
# ax.set_zlabel('tsne-three')
# plt.savefig('outputs/tsnse 3d test diff.png')
# plt.clf()