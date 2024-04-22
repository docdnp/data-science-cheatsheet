# Data Science Cheat Sheet

## Numpy - Key Functions and Examples

| Category                     | Function or Example                    | Description                           |
| ---------------------------- | -------------------------------------- | ------------------------------------- |
| **Creating Arrays**          | `np.array([1, 2, 3])`                  | Create a vector                       |
|                              | `np.array([[1, 2, 3], [4, 5, 6]])`     | Create a 2D array (matrix)            |
| **Array Attributes**         | `arr.ndim`                             | Dimension of an array                 |
|                              | `arr.shape`                            | Shape of an array                     |
|                              | `arr.size`                             | Size (number of elements) of an array |
|                              | `arr.dtype`                            | Data type of elements                 |
| **Array Manipulation**       | `arr.reshape(2, 3)`                    | Reshape arrays                        |
|                              | `arr.ravel()`                          | Flatten arrays                        |
|                              | `np.concatenate([arr1, arr2], axis=0)` | Concatenate arrays                    |
| **Indexing and Slicing**     | `arr[2, 3]` or `arr[:, 1:3]`           | Select elements                       |
|                              | `arr[arr > 5]`                         | Conditional selection                 |
| **Mathematical Ops**         | `np.add(arr1, arr2)` or `arr1 + arr2`  | Element-wise addition                 |
|                              | `np.dot(arr1, arr2)`                   | Matrix multiplication                 |
|                              | `np.sin(arr)`, `np.log(arr)`           | Universal functions (ufuncs)          |
| **Aggregations**             | `arr.sum()`                            | Sum of all elements                   |
|                              | `arr.mean()`                           | Average                               |
|                              | `arr.max()`                            | Maximum                               |
| **Statistical Computations** | `np.median(arr)`                       | Median                                |
|                              | `np.std(arr)`                          | Standard deviation                    |

## Pandas - Key Functions and Examples

Pandas has two basic data structures:

- **DataFrame**: A DataFrame is a two-dimensional, size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). It is essentially a table where each column is a Series.
- **Series**: A Series is a one-dimensional array-like object containing a sequence of values and associated array of data labels, called its index. It represents a single column in a DataFrame.

### Pandas - Dataframes

| Category                 | Function or Example                                                  | Description                                                                                                                     |
| ------------------------ | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Loading Data**         | `pd.read_csv('file.csv')`                                            | Load data from a CSV file                                                                                                       |
|                          | `pd.read_excel('file.xlsx')`                                         | Load data from an Excel file                                                                                                    |
| **Data Viewing**         | `df.head()`                                                          | View the first few rows of the dataframe                                                                                        |
|                          | `df.tail()`                                                          | View the last few rows of the dataframe                                                                                         |
| **Data Selection**       | `df['column_name']`                                                  | Select a column by its name                                                                                                     |
|                          | `df[['col1', 'col2']]`                                               | Select multiple columns                                                                                                         |
|                          | `df.iloc[0]`                                                         | Select a row by integer location                                                                                                |
|                          | `df.loc[conditions]`                                                 | Select rows based on label or a boolean array.                                                                                  |
|                          | `df['column'].unique()`                                              | Return unique values of a column in the form of an array.                                                                       |
| **Filtering Data**       | `df[df['column'] > value]`                                           | Filter rows based on condition                                                                                                  |
| **Missing Data**         | `df.dropna()`                                                        | Drop rows with missing values                                                                                                   |
|                          | `df.fillna(value)`                                                   | Fill missing values with a specified value                                                                                      |
|                          | `df.isnull()`                                                        | Check for missing values in the DataFrame, returning a boolean mask where `True` indicates missing values.                      |
|                          | `df.notnull()`                                                       | Check for non-missing values in the DataFrame, returning a boolean mask where `True` indicates non-missing values.              |
|                          | `df['column'].isna()`                                                | Check for missing values in a specific column, same as `isnull()`.                                                              |
| **Data Manipulation**    | `df.apply(func)`                                                     | Apply a function across the dataframe                                                                                           |
|                          | `df['column'].map(func)`                                             | Apply a mapping function to a column                                                                                            |
| **Grouping Data**        | `df.groupby('column').sum()`                                         | Group data and calculate sum per group                                                                                          |
|                          | `df.groupby('column').mean()`                                        | Group data and calculate average per group                                                                                      |
| **Joining Data**         | `pd.merge(df1, df2, on='key', how='inner')`                          | Merge two dataframes on a key column (inner join)                                                                               |
|                          | `pd.merge(df1, df2, on='key', how='left')`                           | Left outer join                                                                                                                 |
|                          | `pd.merge(df1, df2, on='key', how='right')`                          | Right outer join                                                                                                                |
|                          | `pd.merge(df1, df2, on='key', how='outer')`                          | Full outer join                                                                                                                 |
|                          | `df1.merge(right=df2, on='key', how='<how>')`                        | Alternative: Merge two dataframes on a key column                                                                               |
|                          | `pd.concat([df1, df2], axis=0)`                                      | Concatenate dataframes vertically, stacking one on top of the other.                                                            |
|                          | `pd.concat([df1, df2], axis=1)`                                      | Concatenate dataframes horizontally, appending columns of one dataframe to another.                                             |
| **Statistical Analysis** | `df.describe()`                                                      | Generate descriptive statistics                                                                                                 |
|                          | `df.mode()`                                                          | Calculate the mode(s) for each column in the DataFrame, which represents the most frequently occurring value(s) in each column. |
|                          | `df['column'].mode()`                                                | Calculate the mode of a column                                                                                                  |
|                          | `df['column'].min()`                                                 | Calculate the minimum value of a column.                                                                                        |
|                          | `df['column'].max()`                                                 | Calculate the maximum value of a column.                                                                                        |
|                          | `df['column'].mean()`                                                | Calculate the mean of a column                                                                                                  |
|                          | `df['column'].quantile(q)`                                           | Calculate the quantile of a column, where `q` is a float representing the quantile threshold.                                   |
|                          | `pd.crosstab(index=df['col1'], columns=df['col2'])`                  | Create a cross-tabulation to show the frequency with which certain groups of data appear.                                       |
|                          | `pd.crosstab(col1, col2, rownames=['True'], colnames=['Predicted'])` | Create a cross-tabulation with custom row and column names to compare true conditions versus predicted results.                 |
| **Data Export**          | `df.to_csv('file.csv')`                                              | Export dataframe to a CSV file                                                                                                  |
|                          | `df.to_excel('file.xlsx')`                                           | Export dataframe to an Excel file                                                                                               |
| **DataFrame Metadata**   | `df.info()`                                                          | Print a concise summary of a DataFrame                                                                                          |
|                          | `df.dtypes`                                                          | Get the data types of each column                                                                                               |
|                          | `df.shape`                                                           | Returns a tuple representing the dimensionality (number of rows and columns) of the DataFrame.                                  |
|                          | `df.dtypes`                                                          | Get the data types of each column.                                                                                              |
|                          | `df.select_dtypes(include=[types])`                                  | Select columns in a DataFrame based on their data type. Specify types to include using a list like `['number', 'category']`.    |
|                          | `df.select_dtypes(exclude=[types])`                                  | Exclude columns in a DataFrame based on their data type.                                                                        |


### Pandas - Series

| Category                | Function or Example                     | Description                                                                                              |
| ----------------------- | --------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Series Attributes**   | `series.values`                         | Returns the data of the Series as an array.                                                              |
|                         | `series.index`                          | Returns the index (labels) of the Series.                                                                |
|                         | `series.dtype`                          | Returns the data type of the Series.                                                                     |
| **Series Manipulation** | `series.sort_values()`                  | Sorts the Series in ascending or descending order.                                                       |
|                         | `series.drop_duplicates()`              | Returns a Series with duplicate values removed.                                                          |
|                         | `series.reset_index(drop=True)`         | Resets the index of the Series, making it a simple range index, and optionally drops the previous index. |
| **Series Analysis**     | `series.describe()`                     | Provides a summary of statistics pertaining to the Series data.                                          |
|                         | `series.mean()`                         | Computes the mean of the Series.                                                                         |
|                         | `series.median()`                       | Computes the median of the Series.                                                                       |
|                         | `series.mode()`                         | Computes the mode of the Series.                                                                         |
|                         | `series.quantile([0.25, 0.5, 0.75, 1])` | Computes quantiles for the Series data.                                                                  |

## Sklearn - Key Functions and Examples

| Category                | Function or Example                                    | Description                                                 |
| ----------------------- | ------------------------------------------------------ | ----------------------------------------------------------- |
| **Data Preparation**    | `from sklearn.model_selection import train_test_split` | Split data into train and test sets                         |
|                         | `train_test_split(X, y, test_size=0.2)`                | Example of splitting data                                   |
| **Model Training**      | `from sklearn.linear_model import LinearRegression`    | Import a linear regression model                            |
|                         | `model = LinearRegression()`                           | Create an instance of a linear regression model             |
|                         | `model.fit(X_train, y_train)`                          | Fit the model to the training data                          |
| **Prediction**          | `predictions = model.predict(X_test)`                  | Make predictions using the fitted model                     |
| **Model Evaluation**    | `from sklearn.metrics import mean_squared_error`       | Import the mean squared error metric                        |
|                         | `mean_squared_error(y_test, predictions)`              | Calculate the mean squared error of the model predictions   |
| **Model Persistence**   | `from sklearn.externals import joblib`                 | Import joblib for model saving/loading                      |
|                         | `joblib.dump(model, 'model.pkl')`                      | Save a model to a file                                      |
|                         | `model = joblib.load('model.pkl')`                     | Load a model from a file                                    |
| **Data Transformation** | `from sklearn.preprocessing import StandardScaler`     | Import the StandardScaler for scaling features              |
|                         | `scaler = StandardScaler()`                            | Create an instance of StandardScaler                        |
|                         | `X_scaled = scaler.fit_transform(X)`                   | Scale features to be centered and scaled                    |
| **Clustering**          | `from sklearn.cluster import KMeans`                   | Import KMeans for clustering                                |
|                         | `kmeans = KMeans(n_clusters=3)`                        | Create an instance of KMeans with 3 clusters                |
|                         | `kmeans.fit(X)`                                        | Fit the KMeans model to the data                            |
|                         | `clusters = kmeans.predict(X)`                         | Assign samples to clusters                                  |
| **Classification**      | `from sklearn.ensemble import RandomForestClassifier`  | Import a RandomForestClassifier                             |
|                         | `classifier = RandomForestClassifier()`                | Create an instance of a random forest classifier            |
|                         | `classifier.fit(X_train, y_train)`                     | Fit the classifier to the training data                     |
|                         | `predictions = classifier.predict(X_test)`             | Make predictions on the test data                           |
|                         | `from sklearn.linear_model import LogisticRegression`  | Import the logistic regression model                        |
|                         | `model = LogisticRegression()`                         | Create an instance of a logistic regression model           |
|                         | `model.fit(X_train, y_train)`                          | Fit the logistic regression model to the training data      |
|                         | `predictions = model.predict(X_test)`                  | Make predictions using the fitted logistic regression model |
| **Model Evaluation**    | `from sklearn.metrics import mean_squared_error`       | Import the mean squared error metric                        |
|                         | `mean_squared_error(y_test, predictions)`              | Calculate the mean squared error of the model predictions   |


