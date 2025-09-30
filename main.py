# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import modules for machine learning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

Grc_df = pd.read_csv('Grocery_dataset.csv', sep=',')

'''
1. Read the CSV file in Pandas and create a DataFrame named Grc_df. What is the number of
rows and columns in Grc_df? Print the first 10 and last 10 rows of Grc_df. 
'''

print("\n Executing first task...")
print(f"Number of rows and columns in Grc_df: {Grc_df.shape}")
print(f"First 10 rows of Grc_df: {Grc_df.head(10)}")
print(f"Last 10 rows of Grc_df: {Grc_df.tail(10)}")

'''
2. Are there any null values in the Grc_df? If yes, then in which columns and how many?
Finally, handle these null values using any strategy shown during the labs
'''

print("\n Executing second task...")
print(f"Null values in Grc_df: {Grc_df.isnull().sum()}")
# Item_Weight    818 empty
# Outlet_Size    1439 empty

# Replace empty strings with NaN
Grc_df = Grc_df.replace(r'^\s*$', np.nan, regex=True)

# Fill missing values for Item_Weight column with the median, based on row groups
Grc_df["Item_Weight"] = Grc_df.groupby("Item_Identifier")["Item_Weight"].transform(lambda x: x.fillna(x.median()))

# Fill missing values for Outlet_Size column with the mode, based on row groups
Grc_df['Outlet_Size'] = Grc_df.groupby('Outlet_Identifier')['Outlet_Size'].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 'Medium'))

# Drop rows with any NaN values
Grc_df.dropna(inplace = True)

'''
3. How many unique Outlet Sizes are there in the Grc_df? Which outlet size is maximum, and
which is minimum? 
'''

print("\n Executing third task...")
print(f"Unique Outlet Sizes in Grc_df: {Grc_df['Outlet_Size'].unique()}")
print(f"Maximum Outlet Size in Grc_df: {Grc_df['Outlet_Size'].value_counts().idxmax()}")
print(f"Minimum Outlet Size in Grc_df: {Grc_df['Outlet_Size'].value_counts().idxmin()}")

'''
4.How many unique Item Fat Content types are in the Grc_df? List them. Do you see any
issues with the Item Fat Content types? If yes, then handle this issue. 
'''

print("\n Executing fourth task...")
print(f"Unique Item Fat Content types in Grc_df: {Grc_df['Item_Fat_Content'].unique()}")
Grc_df['Item_Fat_Content'] = Grc_df['Item_Fat_Content'].replace({
    'reg': 'Regular',
    'LF': 'Low Fat',
    'low fat': 'Low Fat'
})

'''
5. Drop the columns having index values of 0, 6 and create a new DataFrame Grc_new_df. 
'''

print("\n Executing fifth task...")
Grc_new_df = Grc_df.drop(Grc_df.columns[[0, 6]], axis=1)
print(f"Columns dropped. New DataFrame Grc_new_df created with shape: {Grc_new_df.shape}")

'''
6. Using different Supermarket type listed in the column Outlet_Type create two different
Dataframes from Grc_new_df. Name these Dataframes as Subtype_1 and Subtype_2. 
'''
print("\n Executing sixth task...")

#Finn Ulike Outlet_Type
unique_outlets = Grc_df['Outlet_type'].unique()
print(f"Unique Outlet Types:{unique_outlets}")

#Opprett to Dataframes
SubType_1 = Grc_new_df[Grc_new_df['Outlet_Type']== unique_outlets[0]]
SubType_2 = Grc_new_df[Grc_new_df['Outlet_Type']== unique_outlets[1]]

print(f"Subtype_1 shape: {SubType_1.shape}")
print(f"Subtype_2 shape: {SubType_2.shape}")

'''
7. Using ("ggplot style") create a (2,1) subplot of a box plot showing 5-point summary of 
the column Item_MRP for Subtype_1 and Subtype_2. Which Outlet Type has a higher median MRP?
Are there any outliers?
'''

print("\n Executing seventh task...")

sns.set(style="ggplot")

fig, axes = plt.subplot(2, 1, figsize=(10, 8))

#Boxplot for Subtype_1
sns.boxplot(x="Item_MRP", data=SubType_1, ax=axes[0])
axes[0].set_title(f"Item_MRP Distrubtion for {unique_outlets[0]}")

#Boxplot for Subtype_2
sns.boxplot(x="Item_MRP", data=SubType_2, ax=axes[1])
axes[1].set_title(f"Item_MRP Distrubtion for {unique_outlets[1]}")

plt.tight_layout()
plt.show()

#Median sjekk
median_1 = SubType_1["Item_MRP"].median()
median_2 = SubType_2["Item_MRP"].median()

print(f"Median Item_MRP for {unique_outlets[0]}:{median_1}")
print(f"Median Item_MRP for {unique_outlets[1]}:{median_2}")

if median_1 > median_2:
    print(f"{unique_outlets[0]} has a higher median Item_MRP.")
else:
    print(f"{unique_outlets[1]} has a higher median Item_MRP.")