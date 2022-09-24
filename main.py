#!/usr/bin/env python
# coding: utf-8

# # Data Gathering:

# In[27]:


import pandas as pd
import numpy as np
import tweepy
import re
import json
import datetime
import os
import seaborn as sns
import warnings
import matplotlib.pyplot as plot
import requests
from scipy import stats
from matplotlib.dates import DateFormatter
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[28]:


#read twitter archive enhanced
archive_df = pd.read_csv('twitter-archive-enhanced.csv')


# In[ ]:


#read image prediction tsv
url ="https://d17h27t6h515a5.cloudfront.net/topher/2017/August/599fd2ad_image-predictions/image-predictions.tsv"
response = requests.get(url)

file_name = url.split('/')[-1]

# Write the file with the context manager with:
with open(file_name, 'wb') as file:
     file.write(response.content)
        
image_predictions_df = pd.read_csv(file_name,sep='\t')


# In[29]:


archive_df.info()


# In[30]:


#get data from twitter
consumer_key = 'pYDY4G9hxxxxxxxxxxxwet'
consumer_secret = 'NvgyxxxxxxxxxxxxxxxxxxnhliAJjgy'
access_token = '13687xxxxxxxxxxxxxxxxxxxxxlbb7'
access_secret = '8MhQxxxxxxxxxxxxxxxxxxxxxxxxxx3Yf0aPUp'
BT='AAAxxxxxxxxxxxxxfZrDNCwmtm3JzkBQokytn'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True) #tweepy.API(auth)


# In[32]:


# if the json file was downloaded then read from it  - to avoid redownloading using the API 
errors = []
i=0
if not os.path.isfile('tweet_json.txt'):
    # create the file and write on it
    with open ('tweet_json.txt', 'w') as file:
        for tweet_id in archive_df['tweet_id']:
            try:
                status = api.get_status(tweet_id, wait_on_rate_limit=True,  wait_on_rate_limit_notify=True, tweet_mode = 'extended')
                json.dump(status._json, file)
                file.write('\n')
                i +=1
                if i%15==0:
                    print(i)
            except Exception as e:
                print("Error on tweet id {}".format(tweet_id) + ";" + str(e))
                errors.append(tweet_id)
else:
    print("tweet_json.txt File exists")


# In[33]:


#check the json file was created 
if not os.path.isfile('tweet_json.txt'):
    print('tweet_json.txt - file not found')
else:
    print('tweet_json.txt - file found')


# In[34]:


#Read the downloaded data from the txt file and load it to a dataframe
tweet_list=[]

with open ('tweet_json.txt','r') as file:
    # here you need to read the file line by line
    for line in file:
        tweet = json.loads(line)
        tweet_id = tweet['id']
        favorite_count = tweet['favorite_count']
        retweet_count = tweet['retweet_count']
        user_count = tweet['user']['followers_count']
            
        tweet_list.append({'tweet_id': tweet_id,
                        'favorite_count': favorite_count,
                        'retweet_count': retweet_count})
        
api_df = pd.DataFrame(tweet_list, columns = ['tweet_id', 'favorite_count', 'retweet_count'])

# check the first three tweets in your list

api_df.sample(10)


# In[36]:


#copy dataframes to preserve the original ones
image_predictions_df_clean =image_predictions_df.copy()
archive_df_clean = archive_df.copy()
api_df_clean = api_df.copy()

image_predictions_df.info()


# In[37]:


image_predictions_df.head()


# In[ ]:





# In[ ]:





# # Assessment Summary

# #### Data Assessment
# - for all the 3 data frames
# 

# In[38]:


archive_df_clean.sample(10)


# In[39]:


archive_df_clean.name.value_counts()
archive_df_clean.query('name == "a"')['text']


# In[40]:


image_predictions_df_clean.sample(10)


# In[ ]:





# In[41]:


api_df_clean.sample(10)


# In[42]:


api_df_clean.info()


# In[43]:


image_predictions_df_clean.info()


# In[44]:


archive_df_clean.name.value_counts()


# In[45]:


image_predictions_df_clean.info()


# ### Quality:
# 
# - Remove re-tweets entries from archive table & image prediction table (validity issue)   (Done)
# - Remove re-replies entries from archive table & image prediction table (validity issue)   (Done)
# - Remove tweets entries that doesn't have images from archive table table(validity issue)            (Done)
# - Fix 3 float numerator rating by re-extracting them from the text                                    (Done)
# - convert timestamp to datetime type                                               (Done) 
# - check for letter a & letter an in dog names (in accurate)                        (Done) 
# - Convert None into NaN in dog stage column                                  (Done)
# - Convert None into NaN in the name column
# - Convert data types for the following column into int instead of float : retweet_count in tweet_archive table (Done)
# - Convert data types for the following column into int instead of float : favorite_count in tweet_archive table (Done)
# - Fix the rating_numerator & rating_denominator of all the 16 rows where rating_denominator != 10 in tweet archive table
# - Drop the row where rating_numerator & rating_denominator = 24/7 as it isn't a real rating
# - Fix outliers issue in tweet_archive table for rating_numerator equal 420 and 1776 (which isn't a dog by the way)
# - Inconsistent captilization for p1,p2,p3 columns in image predcition table 
# 

# ### Tidness:
# - The dogs stages columns should be merged to one column (Done)
# - Remove re-tweets and replies columns from archive table  (Done)
# - Merge api_df with archive_df table and image prediction    (Done)
# - rename p1_dog, p1_conf, to confidance_1, prediction, breed  (Done)
# 
# 

# ## Clean

# ### Tidiness

# ##### Define
# - Merge doggo, floofer,pupper,puppo into a newly created column dog_stage

# ##### Code

# In[ ]:





# In[46]:


archive_df_clean.iloc[:, -4:  ] = archive_df_clean.iloc[:, -4:  ].replace('None','')

archive_df_clean['dog_stage'] = archive_df_clean.doggo + archive_df_clean.floofer + archive_df_clean.pupper + archive_df_clean.puppo

archive_df_clean.drop(columns=['doggo', 'floofer', 'pupper', 'puppo'], inplace=True)


# In[47]:


archive_df_clean.dog_stage.value_counts()


# ##### Test

# In[48]:


archive_df_clean.info()


# ##### Define
# - Merge table api_df columns with tweets archive columns
# - Merge table image prediction columns with tweets archive columns

# ##### Code

# In[49]:


archive_df_clean = pd.merge(archive_df_clean, api_df_clean,
                            on=['tweet_id'], how='left')


# ##### Test

# In[50]:


archive_df_clean.info()


# ### Quality

# ##### Define
# - Remove replies from image prediction table and from tweets archive table
# - Remove re-tweets from image prediction table and from tweets archive table
# - Remove entries in tweets archive tables that doesn't have dog images

# ##### Code

# In[51]:


print ("arch len {} vs image pred len P{}".format(len(archive_df_clean),len(image_predictions_df_clean))) 


# In[52]:


#Get retweet and replies entries 
retweet_entries = archive_df_clean[archive_df_clean.retweeted_status_id.notnull()].tweet_id
#print(retweet_entries)
replies_entries = archive_df_clean[archive_df_clean.in_reply_to_status_id.notnull()].tweet_id


# In[53]:


#image_predictions_df_clean.p1_dog.value_counts()


# In[54]:


#remove replies and re-tweets from image prediction table

image_predictions_df_clean = image_predictions_df_clean.loc[~image_predictions_df_clean.tweet_id.isin(retweet_entries)]

image_predictions_df_clean = image_predictions_df_clean.loc[~image_predictions_df_clean.tweet_id.isin(replies_entries)]

len(image_predictions_df_clean)


# In[55]:


#remove re-tweets and replies from tweets archive table

archive_df_clean = archive_df_clean[archive_df_clean.in_reply_to_status_id.isnull()]

archive_df_clean = archive_df_clean[archive_df_clean.retweeted_status_id .isnull()]
len(archive_df_clean)


# In[56]:


# remove tweets that doesn't have any images
tweets_w_images_id = image_predictions_df_clean['tweet_id']

#tests
#len( list(image_predictions_df_clean.tweet_id.unique()))
#tweets_w_images_id 


archive_df_clean= archive_df_clean.loc[archive_df_clean.tweet_id.isin(tweets_w_images_id)]
len(archive_df_clean)


# ##### Test

# In[57]:


print ("arch len {} vs image pred len {} after removing replies, re-tweets and tweets without images".format(len(archive_df_clean),len(image_predictions_df_clean))) 


# In[58]:


archive_df_clean.dog_stage.value_counts()


# ### Tidness

# ##### Define
# - Rename columns P1_conf, p1 and p1_dog in image predictions table
# - Merge a sliced part of the image predictions table with tweet archive table to form a master archive table with all the related data 

# ##### Code

# In[59]:


# Renaming the dataset columns
#image_predictions_reshaped = image_predictions_df_clean.copy()
cols = ['tweet_id', 'jpg_url', 'img_num', 
       'prediction_1', 'confidence_1', 'breed_1',
       'prediction_2', 'confidence_2', 'breed_2',
       'prediction_3', 'confidence_3', 'breed_3']
image_predictions_df_clean.columns = cols

# Reshaping the dataframe
image_predictions_reshaped = pd.wide_to_long(image_predictions_df_clean, stubnames=['prediction', 'confidence', 'breed'], 
    i=['tweet_id', 'jpg_url', 'img_num'], j='prediction_level', sep="_").reset_index()


# In[60]:


#slice the image prediction table dataset for merging
image_predictions_sliced = image_predictions_df_clean[['tweet_id','prediction_1', 'confidence_1', 'breed_1']]


# In[61]:


tweet_archive_master = pd.merge(archive_df_clean, image_predictions_sliced,
                            on=['tweet_id'], how='left')


# In[62]:


tweet_archive_master.sample(10)


# ##### Test

# In[63]:


image_predictions_df_clean.sample(10)
#image_predictions_reshaped.info()


# ##### Define
# - Remove all the re-tweets and replies related colums from the archive_df_clean table

# In[64]:


list(tweet_archive_master)


# ##### Code
# 

# In[65]:


#archive_df_clean = archive_df_clean.drop(['in_reply_to_status_id','in_reply_to_user_id','retweeted_status_id','retweeted_status_user_id','retweeted_status_timestamp'], axis=1)
tweet_archive_master = tweet_archive_master.drop(['in_reply_to_status_id','in_reply_to_user_id','retweeted_status_id','retweeted_status_user_id','retweeted_status_timestamp'], axis=1)


# ##### Test

# In[66]:


tweet_archive_master


# In[67]:


tweet_archive_master.info()


# ### Quality

# ##### Define
# - Apply str.capitalize to all the 3 columns p1,p2,p3 in image predictions dataset
# 

# ##### Code
# 

# In[ ]:





# In[68]:


image_predictions_df_clean['p1'] = image_predictions_df_clean['p1'].str.capitalize() 
image_predictions_df_clean['p2'] = image_predictions_df_clean['p2'].str.capitalize() 
image_predictions_df_clean['p3'] = image_predictions_df_clean['p3'].str.capitalize() 


# ##### Test

# In[ ]:


image_predictions_df_clean.p1.value_counts() 


# ##### Define
# - convert timestamp into datetime
# 

# ##### Code
# 

# In[ ]:


tweet_archive_master.info()


# In[ ]:


# To datetime
tweet_archive_master.timestamp = pd.to_datetime(tweet_archive_master.timestamp)


# ##### Test

# In[ ]:


tweet_archive_master.info()


# In[ ]:


tweet_archive_master.head()


# ##### Define
# - Re- extract names from text column to fix names called "a" and "an"
# - Convert All None entries into Nan

# ##### Code

# In[ ]:


tweet_archive_master.name.value_counts(dropna = False)


# In[69]:


pattern_2 = re.compile(r'(?:name(?:d)?)\s{1}(?:is\s)?([A-Za-z]+)')
for index, row in tweet_archive_master.iterrows():  
    try:
        if row['name'] == "a":
            c_name = re.findall(pattern_2, row['text'])[0]
            tweet_archive_master.loc[index,'name'] = tweet_archive_master.loc[index,'name'].replace('a', c_name)
        elif row['name'] == 'an':
            c_name = re.findall(pattern_2, row['text'])[0]
            tweet_archive_master.loc[index,'name'] = tweet_archive_master.loc[index,'name'].replace('an', c_name)
    except IndexError:
        tweet_archive_master.loc[index,'name'] = np.nan


#convert All None Names into NaN
tweet_archive_master.loc[tweet_archive_master.name == "None", 'name'] = np.nan


# ##### Test

# In[70]:


tweet_archive_master.name.value_counts(dropna = False)


# ### Define
# 
# - Re-extract the numerator rating from the text column in the archive_df_clean table where the rating is a float value
# - Convert the numerator rating column into a float value

# ### Code

# In[71]:


#the rating before fixing  
tweet_archive_master.rating_numerator.value_counts()


# In[ ]:





# In[72]:


tweet_archive_master['rating_numerator'] = tweet_archive_master.text.str.extract('(\d+\.?\d?\d?)\/\d{1,3}', expand = False).astype('float')


# ### Test

# In[73]:


tweet_archive_master['rating_numerator'].value_counts()


# ### Define
# 
# - Fix the rating_numerator & rating_denominator of all the 16 rows where rating_denominator != 10 in tweet archive table
# - Drop the row where rating_numerator & rating_denominator = 24/7 as it isn't a real rating

# ### Code

# In[74]:



#tweet_archive_master.query('rating_numerator == 84')
#indics = tweet_archive_master.query('rating_denominator != 10').rating_denominator.index

n_rating=[84,144,204,165,121,99,80,60,44,88,45]
d_rating=[70,120,170,150,110,90,80,50,40,80,50]

column_1 = 'rating_numerator'
column_2 = 'rating_denominator'

#print(indics)

for x in range(len(n_rating)):
  #print("{} - {}".format(x,tweet_archive_master.iloc[indics[x]][column_1]))
  
  mask_1 = tweet_archive_master.rating_numerator == n_rating[x]
  mask_2 = tweet_archive_master.rating_denominator == d_rating[x]
  
  #if len(tweet_archive_master.ix[mask_1 & mask_2]):
  ind = tweet_archive_master.ix[mask_1 & mask_2].rating_numerator.index[0]
  #print (indics[x])

  div_by = int (d_rating[x]/10)
  
  #tweet_archive_master.iloc[indics[x]][column_1] = n_rating[x]/div_by
  #tweet_archive_master.iloc[indics[x]][column_2] = int(d_rating[x]/div_by)
 
  tweet_archive_master.loc[mask_1 & mask_2, ['rating_numerator', 'rating_denominator']] = [n_rating[x]/div_by , 10]
  #print(tweet_archive_master.loc[mask_1 & mask_2, ['rating_numerator', 'rating_denominator']])


# In[75]:


#Fixing rating manually

n_rating=[50,4,9,7,1]
d_rating=[50,20,11,11,2]

#correct numerator ratings
cor_num= [11,13,14,10,9]

column_1 = 'rating_numerator'
column_2 = 'rating_denominator'

for x in range(len(n_rating)):

    mask_1 = tweet_archive_master.rating_numerator == n_rating[x]
    mask_2 = tweet_archive_master.rating_denominator == d_rating[x]

    tweet_archive_master.loc[mask_1 & mask_2, ['rating_numerator', 'rating_denominator']] = [cor_num[x] , 10]
    #print(tweet_archive_master.loc[mask_1 & mask_2, ['rating_numerator', 'rating_denominator']])


# In[76]:


#manual remove index 382 with rating 24/7 as it isn't correct 

tweet_archive_master.drop([382],inplace = True)


# In[ ]:





# In[ ]:





# ### Test

# In[77]:


tweet_archive_master.query('rating_denominator != 10').rating_denominator


# In[ ]:





# ##### Define
# 
# - Convert "" into NaN in the four dog_stage column

# ##### Code

# In[78]:


tweet_archive_master.dog_stage = tweet_archive_master.dog_stage.replace('',np.nan) 


# In[79]:


tweet_archive_master.loc[tweet_archive_master.dog_stage == 'doggopupper', 'dog_stage'] = 'doggo-pupper'
tweet_archive_master.loc[tweet_archive_master.dog_stage == 'doggopuppo', 'dog_stage'] = 'doggo-puppo'
tweet_archive_master.loc[tweet_archive_master.dog_stage == 'doggofloofer', 'dog_stage'] = 'doggo-floofer'


# ##### Test

# In[80]:


archive_df_clean.dog_stage.value_counts()


# In[81]:


tweet_archive_master.dog_stage.value_counts(dropna = False)


# In[82]:


tweet_archive_master.info()


# In[83]:


tweet_archive_master.head()


# In[84]:


tweet_archive_master.sample(7)


# In[85]:


tweet_archive_master.tail()


# In[86]:


tweet_archive_master.info()


# In[87]:


tweet_archive_master.retweet_count.value_counts(dropna = False)


# In[88]:


tweet_archive_master.dog_stage.value_counts(dropna = False)


# In[ ]:





# ### Quality

# ##### Define
# - Convert data types for the following column into int instead of float : favorite_count in tweet_archive table

# ##### Code

# In[89]:


tweet_archive_master.favorite_count= tweet_archive_master.favorite_count.fillna(0)


# In[90]:


tweet_archive_master.favorite_count =tweet_archive_master.favorite_count.astype('Int64') 


# In[91]:


tweet_archive_master.sample(4)


# ##### Test

# In[92]:


tweet_archive_master.info()


# ##### Define
# - Convert data types for the following column into int instead of float : retweet_count in tweet_archive table

# ##### Code

# In[93]:


tweet_archive_master.retweet_count= tweet_archive_master.retweet_count.fillna(0)


# In[94]:


tweet_archive_master.retweet_count =tweet_archive_master.retweet_count.astype('Int64') 


# In[95]:


tweet_archive_master.sample(4)


# ##### Test

# In[96]:


tweet_archive_master.info()


# In[97]:


tweet_archive_master.dog_stage.value_counts()


# In[ ]:





# In[ ]:





# In[ ]:





# ##### Define
# - Drop outliers rows in tweet_archive table for rating_numerator equal 420 and 1776 (which isn't a dog by the way)

# ##### Code

# In[ ]:





# In[ ]:





# In[98]:


#remove outliers
index = tweet_archive_master.index
condition = tweet_archive_master["rating_numerator"] == 420
outl1_indices = index[condition]
outl1_indices[0]


# In[99]:


condition = tweet_archive_master["rating_numerator"] == 1776
outl2_indices = index[condition]
outl2_indices[0]


# In[100]:


tweet_archive_master.drop([outl1_indices[0], outl2_indices[0]],inplace = True)


# ##### Test

# In[101]:


tweet_archive_master.rating_numerator.value_counts()


# #### Save Data Frames into CSV files

# In[102]:


tweet_archive_master.to_csv('twitter_archive_master.csv',index =False)


# In[103]:


image_predictions_df_clean.to_csv('image_clean.csv',index =False)


# #### Analyze and visualize Data
# - At least three (3) insights and one (1) visualization 

# In[104]:


df_t = tweet_archive_master.copy()


# In[105]:


tweet_archive_master.info()


# In[106]:


image_predictions_df_clean.info()


# In[82]:


#preparing the data frame for visualization

sns.set_context(context='notebook')
sns.set(rc={'figure.figsize':(10,6)})

#Set the time stamp column as the index of the dataframe, (definitely after making it of type datatime)
df_t.index = df_t['timestamp']

#Drop the column that is now used as an index, as it won’t be of benefit anymore:
df_t.drop(columns='timestamp',  inplace=True)

#Sort the index
df_t.sort_index(inplace=True)


# In[83]:


#Group by the time frame 
data_to_plot = df_t.groupby([(df_t.index.year),(df_t.index.month)]).rating_numerator.mean()
#print(data_to_plot)

#plot your data
data_to_plot.plot(style='-ro', figsize=(12,8),label='Average Dog Rating',color='b')

fig1 = plot.gcf()



plot.xlabel('Tweet Date',fontsize=15)
plot.ylabel('Dogs Rating Out of 10',fontsize=15)
plot.title('Dogs Rating Over Time')
plot.legend()

plot.xticks([0,7,14,21],['Nov,2015', 'May,2016', 'Jan,2017', 'Aug,2017'])

fig1.savefig('Dogs Rating Over Time.png', dpi=100)
plot.show();
plot.draw();


# In[84]:


#Group by the time frame 
data_to_plot = df_t.groupby([(df_t.index.year),(df_t.index.month)]).tweet_id.count()
#print(data_to_plot)

#plot your data
data_to_plot.plot(style='-ro', figsize=(12,8),label='Average Dog Rating',color='b')

fig1 = plot.gcf()



plot.xlabel('Tweet Date',fontsize=15)
plot.ylabel('Tweet Counts',fontsize=15)
plot.title('Twitter Account Activity Over Time')
plot.legend()

plot.xticks([0,7,14,21],['Nov,2015', 'May,2016', 'Jan,2017', 'Aug,2017'])

fig1.savefig('Twitter Account Activity.png', dpi=100)
plot.show();
plot.draw();


# In[ ]:





# In[ ]:





# In[85]:



data_to_plot = df_t.groupby([(df_t.prediction_1)]).tweet_id.count().sort_values(ascending=False)[12::-1]
print(data_to_plot)

#plot your data
data_to_plot.plot(kind="bar")#.plot(style='bar', figsize=(12,8),label='Average rating');

#For saving proper images
fig1 = plot.gcf()


plot.ylabel('Tweet Counts',fontsize=15)
plot.xlabel('Dogs Breeds',fontsize=18)
plot.title('Top rated Dog Breeds')

fig1.savefig('Top rated Dog Breeds.png', dpi=100)
plot.show();
plot.draw();


# In[88]:



data_to_plot = df_t.groupby([(df_t.prediction_1)]).favorite_count.mean().sort_values(ascending=False)[12::-1]
print(data_to_plot)

#plot your data
data_to_plot.plot(kind="bar")#.plot(style='bar', figsize=(12,8),label='Average rating');

#For saving proper images
fig1 = plot.gcf()


plot.ylabel('Average Favorite Tweet',fontsize=15)
plot.xlabel('Dogs Breeds',fontsize=18)
plot.title('Favorite Dog Breeds')

fig1.savefig('Favorite Dog Breeds.png', dpi=100)
plot.show();
plot.draw();


# In[ ]:





# In[89]:


#Group by the time frame you want to use:
data_to_plot = df_t.groupby([(df_t.prediction_1)]).retweet_count.mean().sort_values(ascending=False)[12::-1]
print(data_to_plot)

#plot your data
data_to_plot.plot(kind="bar")#.plot(style='bar', figsize=(12,8),label='Average rating')

fig1 = plot.gcf()

plot.ylabel('Average Retweet Counts',fontsize=15)
plot.xlabel('Dogs Breeds',fontsize=15)
plot.title('Popular Dog Breed Retweets')

fig1.savefig('Popular Dog Breed Retweets.png', dpi=100)
plot.show();
plot.draw();


# In[ ]:





# In[657]:



r_data_to_plot = df_t.groupby([(df_t.index.year),(df_t.index.month)]).retweet_count.mean()
f_data_to_plot = df_t.groupby([(df_t.index.year),(df_t.index.month)]).favorite_count.mean()

#print(r_data_to_plot)
#print(f_data_to_plot)

#plot your data
r_data_to_plot.plot(style='-ro', figsize=(12,8),label='Average Retweets',color='y')
f_data_to_plot.plot(style='-ro', figsize=(12,8),label='Average Favorites',color='b')

fig1 = plot.gcf()

plot.xlabel('Tweet Date',fontsize=15)
plot.ylabel('Average Rating',fontsize=15)
plot.title('Followers Engagment Over Time')
plot.legend()

plot.xticks([0,7,14,21],['Nov,2015', 'May,2016', 'Jan,2017', 'Aug,2017'])

fig1.savefig('Followers Engagment Over Time.png', dpi=100)
plot.show();
plot.draw();


# In[504]:


#scatter plot between retweets & favorite tweet counts
df_t = tweet_archive_master.copy()

#df_t.plot(y='retweet_count',x='favorite_count',kind='scatter')

sns.regplot(y='retweet_count',x='favorite_count', data=df_t, scatter_kws={'alpha':0.2})

fig1 = plot.gcf()

plot.xlabel('Favorite Count',fontsize=15)
plot.ylabel('Retweet Count',fontsize=15)
plot.title('Relation between Favorite Count & Retweet Count')
plot.legend()

#plot.xticks([0,7,14,21],['Nov,2015', 'May,2016', 'Jan,2017', 'Aug,2017'])

fig1.savefig('Relation between Favorite Count & Retweet Count.png', dpi=100)
plot.show();
plot.draw();


# In[507]:


df_t = tweet_archive_master.copy()


r_data_to_plot = df_t.groupby([(df_t.dog_stage)]).retweet_count.mean()
f_data_to_plot = df_t.groupby([(df_t.dog_stage)]).favorite_count.mean()

#print (r_data_to_plot)
ind = np.arange(len(r_data_to_plot))  # the x locations for the groups
width = 0.35       # the width of the bars

# plot bars
red_bars = plot.bar(ind, r_data_to_plot, width, color='b', alpha=.7, label='Average Retweets')
white_bars = plot.bar(ind + width, f_data_to_plot, width, color='tab:blue', alpha=.7, label='Average Favorites')

fig1 = plot.gcf()

# title and labels
plot.ylabel('Average Counts')
plot.xlabel('Dog Stages')
plot.title('Engagment among Different Dog Stages')

plot.rcParams["figure.figsize"]=(18, 12)
locations = ind + width / 2  # xtick locations

labels = [  'doggo','doggo-floofer', 'doggo-pupper', 'doggo-puppo', 'floofer','pupper', 'puppo']  # xtick labels
plot.xticks(locations, labels)

# legend
plot.legend()

fig1.savefig('Engagment among Different Dog Stages.png', dpi=100)
plot.show();
plot.draw();


# In[603]:


df_t = tweet_archive_master.copy()
df_t['breed_1'].value_counts().plot(kind='pie', figsize=(5,5), startangle = 90, wedgeprops = {'width': 0.98},labels=('Dogs', 'Non-Dogs'))

plot.title('Dog vs Non-Dog Predictions',fontsize=16);
plot.ylabel('');

fig1 = plot.gcf()

# legend
plot.legend()

fig1.savefig('Dog vs Non -Dog Predictions.png', dpi=100)
plot.show();
plot.draw();


# #### Reports
# - Create a 300-600 word written report called wrangle_report.pdf or wrangle_report.html that briefly describes your wrangling efforts. This is to be framed as an internal document. 
# 
# - Create a 250-word-minimum written report called act_report.pdf or act_report.html that communicates the insights and displays the visualization(s) produced from your wrangled data. This is to be framed as an external document, like a blog post or magazine article, for example

# In[ ]:


#he ratings probably aren't all correct. Same goes for the dog names and probably dog stages (see below for more information on these) too. You'll need to assess and clean these columns if you want to use them for analysis and visualization.

