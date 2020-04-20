#!/usr/bin/env python
# coding: utf-8

# Must needed setup before every assignment

# In[1]:


get_ipython().system('pip install pyspark')
get_ipython().system('pip install -U -q PyDrive')
get_ipython().system('apt install openjdk-8-jdk-headless -qq')
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"


# Imports

# In[1]:


from math import e, log, ceil
import numpy as np
from numba import jit
from time import time
from math import inf
import shelve
import matplotlib.pyplot as pl


# Jit

# In[2]:


@jit
# http://numba.pydata.org/numba-doc/0.17.0/user/jit.html
def hash_fcn(a, b, x): # p, n_buckets
    return ((a * (x % p) + b) % p) % numBuckets


# Variables

# In[3]:


delta = e ** (-5)
eps = e * 1e-4
p = 123457
aa = []
bb = []
numHashes = ceil(log(1 / delta))
numBuckets = ceil(e / eps)
#fileSuffix = '_tiny'
fileSuffix = ''


# Open files

# In[4]:


with open('hash_params.txt') as f:
    for line in f:
        a, b = line.split()
        aa.append(int(a))
        bb.append(int(b))
assert len(aa) == numHashes

c = np.zeros((numHashes, numBuckets), dtype=int)

t0 = time()
with open('words_stream' + fileSuffix + '.txt') as f:
    for i, wordID in enumerate(f):
        if i % 100000 == 0:
            print(i)
        wordID = int(wordID)
        for j in range(numHashes):
            c[j, hash_fcn(aa[j], bb[j], wordID)] += 1
t = i + 1
print(time() - t0)

realCounts = {}
with open('counts' + fileSuffix + '.txt') as f:
    for line in f:
        wordID, cnt = line.split()
        realCounts[int(wordID)] = int(cnt)


# Code

# In[5]:


hashCounts = {}
for wordID in realCounts:
    hashCount = inf
    for j in range(numHashes):
        cjh = c[j, hash_fcn(aa[j], bb[j], wordID)]
        hashCount = min(hashCount, cjh)
    hashCounts[wordID] = hashCount

error = []
exact = []
for wordID, realCount in realCounts.items():
    error.append((hashCounts[wordID] - realCount) / realCount)
    exact.append(realCount / t)
pl.loglog(exact, error, '.')


my_shelf = shelve.open('allVar' + fileSuffix, 'n') # 'n' for new
for key in dir():
    try:
        my_shelf[key] = globals()[key]
    except: # TypeError:
        # __builtins__, my_shelf, and imported modules can not be shelved.
        print('ERROR shelving: {0}'.format(key))
my_shelf.close()


# Plotting

# In[8]:


fileSuffix = ''

my_shelf = shelve.open('allVar' + fileSuffix)
error = my_shelf['error']
exact = my_shelf['exact']
my_shelf.close()
m_error = min(error)
M_error = max(error)
m_exact = min(exact)
M_exact = max(exact)

pl.loglog(exact, error, '.', markersize=.5)
pl.loglog([m_exact, M_exact], [1, 1], ':', color='xkcd:gray')
pl.loglog([1e-5, 1e-5], [m_error, M_error], ':', color='xkcd:gray')
pl.xlabel('Exact word frequency')
pl.ylabel('Relative error')
pl.savefig('plot.png', dpi=300, bbox_inches='tight')
pl.show()


# In[ ]:





# In[ ]:




