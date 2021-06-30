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
wine = pd.read_csv("wine.csv")
wine


# %%
wine.info()

# %% [markdown]
# ### all numeircal variables |

# %%
wine.isna().sum()

# %% [markdown]
# ### no missing values 
# %% [markdown]
# ## Checking for Distribution of the data 
# 

# %%
a = 5  # number of rows
b = 3  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(15,28))

for i in wine.columns:
    plt.subplot(a, b, c)
    plt.title('{}'.format(i))

    sns.histplot(data= wine, x= i)

    c = c + 1

plt.show()

# %% [markdown]
# ## Numerical variables are usually of 2 type
# ## Continous variable and Discrete Variables

# %%
discrete_feature=[feature for feature in wine.columns if len(wine[feature].unique())<25]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# %%
continuous_feature=[feature for feature in wine.columns if feature not in discrete_feature ]
print("Continuous feature Count {}".format(len(continuous_feature)))


# %%
a = 7  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(13,20))

for i in continuous_feature:
    plt.subplot(a, b, c)

    sns.histplot(x= i ,data= wine, element= "poly", palette="deep" )

    c = c + 1

plt.show()

# %% [markdown]
# ## Checking for outliers

# %%
a = 16  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(13,50))

for i in wine.columns:
    plt.subplot(a, b, c)

    sns.boxplot(x= i ,data= wine, palette="deep" )

    c = c + 1

plt.show()


# %%
for i in wine.columns:
    s = wine[i]
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
for i in wine.columns:
    s = wine[i]
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    iqr_lower = q1 - 1.5 * iqr
    iqr_upper = q3 + 1.5 * iqr

    wine[i] = pd.DataFrame(np.where(wine[i] > iqr_upper, iqr_upper, np.where(wine[i] <  iqr_lower,  iqr_lower, wine[i])))

# %% [markdown]
# ## outliers Removed

# %%
for i in wine.columns:
    s = wine[i]
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
len(wine.columns) # all numerical columns


# %%
wine_normal = scale(wine) # Normalizing the numerical data 
wine_normal


# %%
pca = PCA(n_components = 14)
pca_values = pca.fit_transform(wine_normal)


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
final_wine = pd.concat([wine.Type, pca_data.iloc[:, 0:6]], axis = 1)
final_wine


# %%
# Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = final_wine.comp0, y = final_wine.comp1)

# %% [markdown]
# # Heirarchical-Clustering

# %%

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)


# %%
heir_data_norm = norm_func(final_wine.iloc[:,:])
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
# ## Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram

# %%
h_complete = AgglomerativeClustering(n_clusters = 8, linkage = 'complete', affinity = "euclidean").fit(heir_data_norm) 
h_complete.labels_


# %%
cluster_labels = pd.Series(h_complete.labels_)
final_wine['clust'] = cluster_labels # creating a new column and assigning it to new column


# %%
final_wine.iloc[:, :].groupby(final_wine.clust).mean()

# %% [markdown]
# ## Plot of Heirarichical data clustering 

# %%
a = 6  # number of rows
b = 2  # number of columns
c = 1  # initialize plot counter

fig = plt.figure(figsize=(13,30))

for i in final_wine.columns:
    plt.subplot(a, b, c)

    sns.histplot(x= i,data= final_wine, hue= "clust",palette="deep", element= "poly" )

    c = c + 1

plt.show()


# %%



