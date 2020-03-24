
# coding: utf-8

# Select columns not JSON encoded

# In[29]:


import pandas as pd
import numpy as np

kcks = pd.read_csv("../data/joined.csv")
print(f'Original shape: {kcks.shape}')
print(f'Original columns: {kcks.columns}')

# Remove columns and rows with missing information
kcks.drop(["friends", "is_backing", "is_starred", "permissions"], axis=1, inplace=True)
print(f'Shape after drop empty columns: {kcks.shape}') 
kcks.dropna( axis=0, how='any', inplace=True)
kcks.reset_index(drop=True, inplace=True)
print(f'Shape after drop rows with na: {kcks.shape}')

print(f'Shape after drop rows with na: {kcks.shape}')
#print(f'Tail: {kcks.tail()}')


# In[30]:


# Columns in JSON format are: category, creator, location, photo, profile, urls. They are treated separately for now.

# Drop the JSON format columns into a new df
no_json_df = kcks.drop(['category', 'creator', 'location', 'photo', 'profile', 'urls'], axis=1)
no_json_df.head()


# Drop country and country_displayable_name because they are redundant with varaibles extracted from the JSON encoded columns.
# Drop currency_symbol

# In[32]:


no_json_df.drop(['country', 'country_displayable_name', 'currency_symbol'], axis=1, inplace=True)
no_json_df.iloc[:,0:10].head()


# Drop source_url	

# In[33]:


no_json_df.drop(['source_url'], axis=1, inplace=True)
no_json_df.iloc[:,10:20].head()


# In[34]:


no_json_df.iloc[:,20:30].head()


# Convert unix dates to datetime in 'created_at', 'deadline', 'launched_at', 'state_changed_at'

# In[35]:


no_json_df.columns


# In[37]:


date_cols = ['created_at', 'deadline', 'launched_at', 'state_changed_at']

for column in date_cols:
        no_json_df[column] = pd.to_datetime(no_json_df[column],yearfirst=True, unit='s').dt.normalize() #normalize removes time


# In[38]:


no_json_df.iloc[:,0:15].head()


# In[27]:


no_json_df.iloc[:,15:25].head()


# Weekday columns for each date variable

# In[48]:


date_cols = ['created_at', 'deadline', 'launched_at', 'state_changed_at']

for column in date_cols:
    new = "weekday_" + no_json_df[column].name 
    no_json_df[new] = no_json_df[column].dt.weekday
print(no_json_df.columns)
no_json_df.iloc[0:4,20:30]


#  Month columns for each date variable

# In[50]:


date_cols = ['created_at', 'deadline', 'launched_at', 'state_changed_at']

for column in date_cols:
    new = "month_" + no_json_df[column].name 
    no_json_df[new] = no_json_df[column].dt.month
print(no_json_df.columns)
no_json_df.iloc[0:4,27:40]


# Year columns for each date variable

# In[51]:


date_cols = ['created_at', 'deadline', 'launched_at', 'state_changed_at']

for column in date_cols:
    new = "year_" + no_json_df[column].name 
    no_json_df[new] = no_json_df[column].dt.year
print(no_json_df.columns)
no_json_df.iloc[0:4,30:45]


# Compute initial_found_rising_duration

# In[63]:


no_json_df["initial_found_rising_duration"] = (no_json_df['deadline']- no_json_df["launched_at"])/np.timedelta64(1,'D')

no_json_df.iloc[0:4,35:40]


# In[ ]:


Compute found_rising_duration


# In[64]:


no_json_df["found_rising_duration"] = (no_json_df['state_changed_at']- no_json_df["launched_at"])/np.timedelta64(1,'D')

no_json_df.iloc[0:4,35:40]


# In[65]:


print(no_json_df.columns)


# In[67]:


no_json_df.to_hdf("../data/no_json_df_dates_variables.h5", key = 'dates_variables')

