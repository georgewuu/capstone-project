#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
N_num = 17775132
np.random.seed(N_num)


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/george/Downloads/spotify52kData.csv')
data


# In[4]:


#Q1
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
fig, axs = plt.subplots(2, 5, figsize=(15, 6))

for i, feature in enumerate(features):
    row = i //5
    col = i % 5
    axs[row, col].hist(data[feature], bins = 20, alpha = 0.5)
    axs[row, col].set_title(feature)

plt.show()


# In[6]:


#Q2
plt.figure(figsize = (10,6))
plt.scatter(data['duration'], data['popularity'])
plt.title('scatterplot of relationship between duration and popularity')
plt.xlabel('duration(ms)')
plt.ylabel('popularity')
plt.grid(True)
plt.show()

correlation_coefficient = data['duration'].corr(data['popularity'])

print('correlation coefficient:', correlation_coefficient)


# In[205]:


#Q3
fig, axs = plt.subplots(figsize=(15, 6))
axs.hist(data['popularity'], bins = 30)
plt.show()


# In[199]:


#Q3
from scipy.stats import mannwhitneyu
exp_songs = data[data['explicit'] == True]['popularity'].dropna()
non_exp_songs = data[data['explicit'] == False]['popularity'].dropna()

u_stats, p_value = mannwhitneyu(exp_songs, non_exp_songs)

alpha = 0.05 #significance level
median_exp = exp_songs.median()
median_non_exp = non_exp_songs.median()

if p_value < alpha:
    print(f'{p_value} indicates a significant difference in popularity between explicit and non explicit songs.')
    if median_exp > median_non_exp:
        print("Explicit songs are more popular")
    else:
        print("non explicit songs are more popular")
else:
    print(f'{p_value} does not indicate a significant difference in popularity between explicit and non explicit songs.')


# In[143]:


#Q4
major_songs = data[data['mode'] == 1]['popularity'].dropna()
minor_songs = data[data['mode'] == 0]['popularity'].dropna()

u_stats, p_value = mannwhitneyu(major_songs, minor_songs)

alpha = 0.05 #significance level
median_major = major_songs.median()
median_minor = minor_songs.median()

if p_value < alpha:
    print(f'{p_value} indicates a significant difference in popularity between major and minor songs.')
    if median_major > median_minor:
        print("Major songs are more popular on average.")
    else:
        print("Minor songs are more popular on average.")
else:
    print(f'{p_value} does not indicate a significant difference in popularity between major and minor songs.')


# In[7]:


#Q5
plt.figure(figsize = (10,6))
plt.scatter(data['energy'], data['loudness'])
plt.title('scatterplot of relationship between energy and loudness')
plt.xlabel('energy')
plt.ylabel('loudness (dB)')
plt.grid(True)
plt.show()

correlation_coefficient = data['energy'].corr(data['loudness'])

print('correlation coefficient:', correlation_coefficient)


# In[145]:


#Q6
results = {}
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

for feature in features:
    X = data[[feature]]
    y = data['popularity']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[feature] = mse
    
best_result = min(results, key = results.get)
print('best result is: ', best_result, '( MSE =', results[best_result],')')


# In[16]:


#6
coefs = {}
for feature in features:
    X = data[feature]
    y = data['popularity']
    
    correlation_coefficient = X.corr(y)
    coefs[feature] = correlation_coefficient

coefs


# In[146]:


#Q7
features = ['duration', 'danceability', 'energy', 'loudness', 'speechiness',
'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
fig, axs = plt.subplots(2, 5, figsize=(15, 6))

for i, feature in enumerate(features):
    row = i //5
    col = i % 5
    axs[row, col].scatter(data[feature], data['popularity'])
    axs[row, col].set_title(feature)

plt.show()


# In[147]:


#Q7
from sklearn.ensemble import RandomForestRegressor

X = data[features]
y = data['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

reg = RandomForestRegressor(n_estimators = 20)
reg.fit(X_train, y_train)

dt_pred = reg.predict(X_test)

dt_mse = mean_squared_error(y_test, dt_pred)

dt_mse


# In[148]:


#Q8
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

prop_explained_var = 0
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[features])

pca = PCA()
transform_data = pca.fit_transform(scaled_data)
variance_ratios = pca.explained_variance_ratio_
for i in range(0, 7):
    prop_explained_var += pca.explained_variance_ratio_[i]
    
#prop_explained_var = pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1] + pca.explained_variance_ratio_[2]
evalues = pca.explained_variance_

#using kaiser criterion, we get 3 principal components
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(evalues) + 1), evalues, align='center', label='Individual explained variance')
plt.ylabel('Eigenvalues')
plt.xlabel('Principal Component (PC)')
plt.title('Scree Plot')
plt.grid(True)
plt.show()

print(f'proportion of variance explained: {prop_explained_var}')


# In[207]:


#Q8
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#transform data using three principal components
pca_6 = PCA(n_components = 6)
X_pca = pca_6.fit_transform(scaled_data)
X_pca

silhouette_scores = []


K = range(2, 6)
for k in K:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(X_pca)
    score = silhouette_score(X_pca, kmeans.labels_)
    silhouette_scores.append(score)
    
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'o--')
plt.xlabel('k (Number of clusters)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.show()
    


# In[17]:


#9
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X_train,X_test, y_train, y_test = train_test_split(data['valence'].values.reshape(-1, 1), data['mode'].values, test_size = 0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r_squared = r2_score(y_test, y_pred)

print(r_squared)


# In[19]:


#10
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

label_encoder = LabelEncoder()
data['genre_label'] = label_encoder.fit_transform(data['track_genre'])

X = data[features]
y = data['genre_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

y_pred = tree_model.predict(X_test)
print(classification_report(y_test, y_pred, target_names = label_encoder.classes_))


# In[ ]:





# In[ ]:




