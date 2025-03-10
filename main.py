import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#function to seperate test cases and training sets

# def shuffle_and_split_data(data, test_ratio) :
#     shuffled_indices = np.random.permutation(len(data)) #it will create an array of positions for data
#     test_set_size = int(len(data) * test_ratio) # determine the no of rows for test case
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]



#downloading and preparing data

file_1 = r"C:\Users\gulsh\OneDrive\Desktop\projects\house-price-prediction-model\housing.csv"
housing = pd.read_csv(file_1)



#retrieving information

# print(housing.head()) # displays firt 5 rows
# print(housing.info()) # displays information about not null values , no of rows etc. 
# print(housing["ocean_proximity"].value_counts()) # counts the differnt values in ocean_proximity column 
# print(housing.describe()) # describe the mathematical terms in the data



#plotting graph

# housing.hist(bins=50, figsize=(12,8))
# plt.show()



# using the function for splitting data into test case and train set

# train_set , test_set = shuffle_and_split_data(housing, 0.2)
# print("no of rows in train set = " ,len(train_set))
# print("no of rows in test set is = " ,len(test_set))



#visualizing geographical data

# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
# plt.show()

# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"]/100, label="population", c="median_house_value", cmap="jet", colorbar=True, legend=True, sharex=False, figsize=(10,7))
# plt.show()



# finding correlations

# using standard correlation coefficient :
# corr_matrix = housing.select_dtypes(include=[np.number]).corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

# another way

# from pandas.plotting import scatter_matrix
# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12,8))
# plt.show()

# the most promising factor is median_house_value and median_house_income

# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid =True)
# plt.show()



# attribute combinations

housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

# corr_matrix2 = housing.select_dtypes(include=[np.number]).corr()
# print(corr_matrix2["median_house_value"].sort_values(ascending=False))



# preparing data for the ML algorithms

# clean the data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median") # creating an instance
housing_num = housing.select_dtypes(include=[np.number]) # selecting only number attributes
imputer.fit(housing_num) # training the instance
X = imputer.transform(housing_num) # replacing values

housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index) # changing numpy arrays to pandas dataframes



# handling categorical and text attributes

housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(8))