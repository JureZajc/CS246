# -*- coding: utf-8 -*-
"""CS246-1aPy2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UioPOjI68H7A7SUXjkUS53Dzow_32CgS
"""

!pip install pyspark
!pip install -U -q PyDrive
!apt install openjdk-8-jdk-headless -qq
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

"""Imports"""

from pyspark import SparkContext, SparkConf
from collections import Counter
import itertools
from pyspark.sql import *
from pyspark.sql.functions import *

"""Setup Spark"""

# create the Spark Session
spark = SparkSession.builder.getOrCreate()

# create the Spark Context
sc = spark.sparkContext

"""Code variables"""

fileName = 'soc-LiveJournal1Adj.txt'
N = 10  #only ouput 10 most possible friends

"""Function"""

def connecteds_and_commons(line):
  minimum = -9999999999
  user, friends = line.split('\t')
  friends = friends.split(',')
  connecteds = [((user, friend), minimum) for friend in friends]
  commons = [(pair, 1) for pair in itertools.permutations(friends, 2)]
  return connecteds + commons

"""Output"""

friendsListRDD = (sc
                  .textFile( fileName, 16 )
                  .flatMap( connecteds_and_commons )
                  .reduceByKey( lambda total, current: total + current )
                  .filter(lambda (pair, counts): counts > 0)
                  .map(lambda ((user, friend), counts): (user, (counts, friend)))
                  .groupByKey()
                  .map(lambda (user, suggestions):(user, Counter( dict( (friend, count) for count, friend in suggestions ) ).most_common( N ) ) )
                  #.cache()
                   )


print "924"
print friendsListRDD.lookup('924')
print "8941"
print friendsListRDD.lookup('8941')
print "8942"
print friendsListRDD.lookup('8942')
print "9019"
print friendsListRDD.lookup('9019')
print "9020"
print friendsListRDD.lookup('9020')
print "9021"
print friendsListRDD.lookup('9021')
print "9022"
print friendsListRDD.lookup('9022')
print "9990"
print friendsListRDD.lookup('9990')
print "9992"
print friendsListRDD.lookup('9992')
print "9993"
print friendsListRDD.lookup('9993')
#print friendsListRDD.collect()