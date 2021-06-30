# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings
get_ipython().run_line_magic('matplotlib', 'inline')


# %%

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch


# %%
Heart_disease = pd.read_csv("heart disease.csv")
Heart_disease


# %%
Heart_disease.info()

# %% [markdown]
# ## All of the data is numerical 

# %%
Heart_disease.isna().sum()

# %% [markdown]
# ### No missing values Detected
# %% [markdown]
# ## Checking for Distribution of the data 
# 

# %%
a = 5  # number of rows
b = 3  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(15,28))

for i in Heart_disease.columns:
    plt.subplot(a, b, c)
    plt.title('{}'.format(i))

    sns.histplot(data= Heart_disease, x= i)

    c = c + 1

plt.show()

# %% [markdown]
# ## Numerical variables are usually of 2 type
# ## Continous variable and Discrete Variables
# %% [markdown]
# ## decreate values

# %%
discrete_feature=[feature for feature in Heart_disease.columns if len(Heart_disease[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))

# %% [markdown]
# ## Continous Features

# %%
continuous_feature=[feature for feature in Heart_disease.columns if feature not in discrete_feature ]
print("Continuous feature Count {}".format(len(continuous_feature)))


# %%
a = 5  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(13,20))

for i in continuous_feature:
    plt.subplot(a, b, c)

    sns.histplot(x= i ,data= Heart_disease, element= "poly", palette="deep" )

    c = c + 1

plt.show()

# %% [markdown]
# ## Checking for outliers

# %%
a = 16  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(13,50))

for i in Heart_disease.columns:
    plt.subplot(a, b, c)

    sns.boxplot(x= i ,data= Heart_disease, palette="deep" )

    c = c + 1

plt.show()


# %%
for i in Heart_disease.columns:
    s = Heart_disease[i]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    iqr_lower = q1 - 1.5 * iqr
    iqr_upper = q3 + 1.5 * iqr
    outliers = dict(s[(s < iqr_lower) | (s > iqr_upper)])

    print(f"Details of {i} \n", "IQR = ", iqr, "\n", "IQR lower ", iqr_lower, "\n" , "IQR upper ",iqr_upper, "\n" ,"outliers = ", outliers, "\n"
         
    )

# %% [markdown]
# ## Replacing outlier with upper and lower limit

# %%
for i in Heart_disease.columns:
    s = Heart_disease[i]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    iqr_lower = q1 - 1.5 * iqr
    iqr_upper = q3 + 1.5 * iqr

    Heart_disease[i] = pd.DataFrame(np.where(Heart_disease[i] > iqr_upper, iqr_upper, np.where(Heart_disease[i] <  iqr_lower,  iqr_lower, Heart_disease[i])))

# %% [markdown]
# ## outliers Removed

# %%
for i in Heart_disease.columns:
    s = Heart_disease[i]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    iqr_lower = q1 - 1.5 * iqr
    iqr_upper = q3 + 1.5 * iqr
    outliers = dict(s[(s < iqr_lower) | (s > iqr_upper)])

    print(f"Details of {i} \n", "IQR = ", iqr, "\n", "IQR lower ", iqr_lower, "\n" , "IQR upper ",iqr_upper, "\n" ,"outliers = ", outliers, "\n"
         
    )

# %% [markdown]
# # PCA

# %%
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


# %%
len(Heart_disease.columns) # all numerical columns


# %%
Heart_disease_normal = scale(Heart_disease) # Normalizing the numerical data 
Heart_disease_normal


# %%
pca = PCA(n_components = 14)
pca_values = pca.fit_transform(Heart_disease_normal)


# %%
# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var


# %%
# Cumulative variance 

var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1


# %%

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")


# %%
# PCA scores
pca_values


# %%
pca_data = pd.DataFrame(pca_values)
pca_data.columns = ['comp0', 'comp1', 'comp2', 'comp3', 'comp4', 'comp5', 'comp6', 'comp7', 'comp8', 'comp9', 'comp10', 'comp11', 'comp12', 'comp13']


# %%
final_Heart_disease = pd.concat([Heart_disease.age, pca_data.iloc[:, 0:10]], axis = 1)
final_Heart_disease


# %%
# Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = final_Heart_disease.comp0, y = final_Heart_disease.comp1)

# %% [markdown]
# # Heirarchical-Clustering

# %%

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


# %%
heir_data_norm = norm_func(final_Heart_disease.iloc[:,:])
heir_data_norm.describe()


# %%
heir_data_norm.isna().sum()


# %%
z = linkage( heir_data_norm, method = "complete", metric = "euclidean")


# %%
# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 90,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# %%
from sklearn.cluster import AgglomerativeClustering

# %% [markdown]
# ## Now applying AgglomerativeClustering choosing 8 as clusters from the above dendrogram

# %%
h_complete = AgglomerativeClustering(n_clusters = 8, linkage = 'complete', affinity = "euclidean").fit(heir_data_norm) 
h_complete.labels_


# %%
cluster_labels = pd.Series(h_complete.labels_)
final_Heart_disease['clust'] = cluster_labels # creating a new column and assigning it to new column


# %%
final_Heart_disease.iloc[:, 2:].groupby(final_Heart_disease.clust).mean()

# %% [markdown]
# ## Plot of Heirarichical data clustering 

# %%
a = 6  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(13,30))

for i in final_Heart_disease.columns:
    plt.subplot(a, b, c)

    sns.histplot(x= i,data= final_Heart_disease, hue= "clust",palette="deep", element= "poly" )

    c = c + 1

plt.show()


# %%



