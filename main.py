#!/usr/bin/env python
# coding: utf-8

# This is a transactional data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
# 
# | Variable Name | Role | Type | Description | Units | Missing Values |
# | :--- | :--- | :--- | :--- | :--- | :--- |
# | `invoice_number` | ID | Categorical | a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation | | no |
# | `stock_code` | ID | Categorical | a 5-digit integral number uniquely assigned to each distinct product | | no |
# | `description` | Feature | Categorical | product name | | no |
# | `quantity` | Feature | Integer | the quantities of each product (item) per transaction | | no |
# | `invoice_date` | Feature | Date | the day and time when each transaction was generated | | no |
# | `unit_price` | Feature | Continuous | product price per unit | sterling | no |
# | `customer_id` | Feature | Categorical | a 5-digit integral number uniquely assigned to each customer | | no |
# | `country` | Feature | Categorical | the name of the country where each customer resides | | no |
# 

# In[1]:


get_ipython().system('uv add scikit-learn pandas numpy matplotlib seaborn mlxtend')


# In[269]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

sns.set_theme(style='whitegrid', palette='pastel')


# In[152]:


df = pd.read_csv("dataset_online_retail.csv")


# In[153]:


df = df.rename(columns={
    'InvoiceNo': 'invoice_number',
    'StockCode': 'stock_code',
    'Description': 'description',
    'Quantity': 'quantity',
    'InvoiceDate': 'invoice_date',
    'UnitPrice': 'unit_price',
    'CustomerID': 'customer_id',
    'Country': 'country',
})
df.info()


# In[154]:


df


# # Preprocessing

# In[155]:


print(f"Null values count:\n{df.isna().sum()}")


# In[156]:


def describe_column(df: pd.DataFrame, column_name: str):
    print(f"Column {column_name}:\nNull values:\t{df[column_name].isna().sum()}\nType:\t\t{df[column_name].dtype}")


# In[157]:


if df['unit_price'].dtype == 'str':
    # Pandas can only convert float values with '.' as decimal separator
    corrected_decimal_separator_unit_prices = df['unit_price'].str.replace(',', '.')

    df['unit_price'] = pd.to_numeric(corrected_decimal_separator_unit_prices, errors='coerce')

describe_column(df, 'unit_price')


# In[236]:


if df['invoice_date'].dtype == 'str':
    df['invoice_date'] = pd.to_datetime(df['invoice_date'], format="%d/%m/%Y %H:%M", errors='coerce')

describe_column(df, 'invoice_date')


# In[159]:


# none of the customer_id values have decimals
if df['customer_id'].dtype == 'float64':
    df['customer_id'] = df['customer_id'].astype('Int64')

describe_column(df, 'customer_id')


# # RFM indicator

# In[285]:


# auxiliary rfm functions
def build_rfm_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_with_invoice_total = df.copy()
    df_with_invoice_total['invoice_total'] = df['unit_price'] * df['quantity']

    result = df_with_invoice_total.groupby('customer_id', as_index=True).agg(
        last_invoice_date=('invoice_date', 'max'),
        recency=('invoice_date', lambda date: (max_invoice_date - date.max()).days),
        frequency=('invoice_date', 'count'),
        monetary=('invoice_total', 'sum')
    )

    result['rfm_score'] = get_rfm_score(result)

    result['customer_segment'] = pd.cut(
        result['rfm_score'],
        bins=[0, 1.6, 3, 4, 5],
        labels=["Lost customer", "Low value customer", "Medium value customer", "High value customer"]
    )

    return result

def get_rfm_score(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame()

    result['recency_rank'] = df['recency'].rank(ascending=False).astype(int)
    result['frequency_rank'] = df['frequency'].rank(ascending=True).astype(int)
    result['monetary_rank'] = df['monetary'].rank(ascending=False).astype(int)

    result['recency_rank_norm'] = (result['recency_rank'] / result['recency_rank'].max()) * 100
    result['frequency_rank_norm'] = (result['frequency_rank'] / result['frequency_rank'].max()) * 100
    result['monetary_rank_norm'] = (result['monetary_rank'] / result['monetary_rank'].max()) * 100

    result.drop(columns=['recency_rank', 'frequency_rank', 'monetary_rank'], inplace=True)

    RFM_RECENCY_WEIGHT = 0.15
    RFM_FREQUENCY_WEIGHT = 0.28
    RFM_MONETARY_WEIGHT = 0.57

    result['rfm_score'] = (
        RFM_RECENCY_WEIGHT * result['recency_rank_norm'] + 
        RFM_FREQUENCY_WEIGHT * result['frequency_rank_norm'] + 
        RFM_MONETARY_WEIGHT * result['monetary_rank_norm']
    ) * 0.05

    return result['rfm_score'].round(2)


# In[295]:


rfm_df = build_rfm_dataframe(df)
rfm_df.describe()


# In[296]:


plt.figure(figsize=(10, 6))

ax = sns.countplot(
    data=rfm_df,
    x='customer_segment', 
    order=["Lost customer", "Low value customer", "Medium value customer", "High value customer"],
    stat="percent"
)

ax.bar_label(ax.containers[0], fmt="%.1f%%")

plt.title('RFM customer segments')
plt.xlabel('Segment')
plt.ylabel('Percentage')

plt.show()


# # PCA analysis

# In[307]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# In[324]:


rfm_df_scaled = StandardScaler().fit_transform(rfm_df[['recency', 'frequency', 'monetary']])

pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_df_scaled)

print('Explained variance ratio per component:', pca.explained_variance_ratio_.round(3))
print('Total explained variance:', pca.explained_variance_ratio_.sum().round(3))

plt.scatter(rfm_pca[:,0], rfm_pca[:,1], s=80)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA scatter plot from RFM analysis')
plt.show()


# # KMeans clustering

# In[305]:


from sklearn.cluster import KMeans


# In[367]:


k_range = range(1, 10)


# ## Elbow method

# In[369]:


inertia_values = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(rfm_pca)

    inertia_values.append(kmeans.inertia_)

chosen_k = 3

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia_values, marker='o', linestyle='-', color='b')

for k, e in zip(k_range, inertia_values):
    plt.annotate(f'{e:.0f}', (k, e), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=9)

# highlight chosen k
plt.scatter(chosen_k, inertia_values[chosen_k - 1], color='red', s=50, zorder=2)

plt.title('Elbow method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.xticks(k_range)
plt.grid(True)
plt.show()


# ## Silhouette score

# In[352]:


from sklearn.metrics import silhouette_score


# In[374]:


for k in range(2, 11):
    cluster = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(rfm_pca)
    print(f'k: {k}\tsilhouette:\t{silhouette_score(rfm_pca, cluster):.3f}')


# In[377]:


pca_df = pd.DataFrame(data=rfm_pca, columns=['pc1', 'pc2'])

kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
pca_df['cluster'] = kmeans.fit_predict(rfm_pca)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    pca_df['pc1'], 
    pca_df['pc2'], 
    c=pca_df['cluster'], 
    cmap='Accent', 
    alpha=0.6
)

plt.title('KMeans from PCA scatter')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# In[ ]:




