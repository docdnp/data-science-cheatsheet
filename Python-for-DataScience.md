# Data Science Cheat Sheet

## Numpy - Key Functions and Examples

### Numpy - Basics

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

### Numpy - Key Functions for `np.random`

| Category                     | Function or Example                          | Description                                                                                               |
| ---------------------------- | -------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Normal Distribution**      | `np.random.normal(loc=0, scale=1, size=100)` | Draw random samples from a normal (Gaussian) distribution with mean `loc` and standard deviation `scale`. |
| **Uniform Distribution**     | `np.random.uniform(low=0, high=1, size=100)` | Draw random samples from a uniform distribution over the half-open interval `[low, high)`.                |
| **Binomial Distribution**    | `np.random.binomial(n=10, p=0.5, size=100)`  | Draw random samples from a binomial distribution with number of trials `n` and success probability `p`.   |
| **Poisson Distribution**     | `np.random.poisson(lam=1.0, size=100)`       | Draw random samples from a Poisson distribution with expected number of occurrences `lam`.                |
| **Exponential Distribution** | `np.random.exponential(scale=1.0, size=100)` | Draw random samples from an exponential distribution with scale parameter `scale`.                        |
| **Integers**                 | `np.random.randint(low=0, high=10, size=5)`  | Draw random integers from the “discrete uniform” distribution in the interval `[low, high)`.              |
| **Random Sampling**          | `np.random.choice(['a', 'b', 'c'], size=10)` | Generate a random sample from a given 1-D array or list.                                                  |
| **Shuffle and Permutations** | `np.random.shuffle(x)`                       | Modify a sequence in-place by shuffling its contents.                                                     |
|                              | `np.random.permutation(x)`                   | Randomly permute a sequence, or return a permuted range.                                                  |
| **Random State Control**     | `np.random.seed(seed=42)`                    | Set the seed of the random number generator for numpy to ensure reproducibility of random operations.     |

Dieser Eintrag stellt sicher, dass die Tabelle die Funktion zur Kontrolle des Zufallszustandes umfasst, was besonders wichtig ist, um reproduzierbare Ergebnisse bei der Verwendung von Zufallsfunktionen zu gewährleisten.
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
|                          | `df.drop('column', axis=1)`                                          | Remove a column from the DataFrame.                                                                                             |
|                          | `df.drop(['column1', 'column2'], axis=1)`                            | Remove multiple columns from the DataFrame.                                                                                     |
|                          | `df.drop(0, axis=0)`                                                 | Remove a row from the DataFrame by index.                                                                                       |
|                          | `df.drop([0, 1], axis=0)`                                            | Remove multiple rows from the DataFrame by index.                                                                               |
|                          | `df.drop(df[df['column'] > value]].index)`                           | Remove all rows that match a given condition.                                                                                   |
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

## Seaborn - Key Functions and Examples

| Category               | Function or Example                                        | Description                                                          |
| ---------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------- |
| **Distribution Plots** | `sns.distplot(data)`                                       | Plot a univariate distribution of observations.                      |
|                        | `sns.kdeplot(data, shade=True)`                            | Plot a kernel density estimate.                                      |
|                        | `sns.histplot(data, kde=True)`                             | Plot histogram with optional kernel density estimate.                |
| **Categorical Plots**  | `sns.boxplot(x='x_col', y='y_col', data=df)`               | Draw a box plot to show distributions with respect to categories.    |
|                        | `sns.violinplot(x='x_col', y='y_col', data=df)`            | Draw a combination of boxplot and kernel density estimate.           |
|                        | `sns.barplot(x='x_col', y='y_col', data=df)`               | Show point estimates and confidence intervals as rectangular bars.   |
| **Scatter Plots**      | `sns.scatterplot(x='x_col', y='y_col', data=df)`           | Plot data and a linear regression model fit.                         |
|                        | `sns.pairplot(data=df)`                                    | Plot pairwise relationships in a dataset.                            |
| **Heatmaps**           | `sns.heatmap(data=matrix, annot=True)`                     | Plot rectangular data as a color-encoded matrix.                     |
| **Regression Plots**   | `sns.regplot(x='x_col', y='y_col', data=df)`               | Plot data and a linear regression model fit.                         |
|                        | `sns.lmplot(x='x_col', y='y_col', data=df)`                | Plot data and regression model fits across a FacetGrid.              |
| **Time Series Plots**  | `sns.lineplot(x='time', y='value', data=df)`               | Draw a line plot with possibility for several semantic groupings.    |
| **Facet Grid**         | `sns.FacetGrid(df, col='col_name', row='row_name')`        | Multi-plot grid for plotting conditional relationships.              |
|                        | `g.map(sns.histplot, 'column_name')`                       | Apply a plotting function to each facet’s subset of the data.        |
| **Cluster Map**        | `sns.clustermap(data=matrix)`                              | Organize and plot a heatmap with hierarchical clustering.            |
| **Joint Plot**         | `sns.jointplot(x='x_col', y='y_col', data=df, kind='hex')` | Plot a bivariate histogram and scatterplot with marginal histograms. |

Hier ist eine Tabelle, die eine Auswahl nützlicher Funktionen der Bibliothek `statsmodels.api` umfasst, geeignet für statistische Modellierung und Analyse. Diese Tabelle kategorisiert die Funktionen nach ihrem Hauptanwendungsbereich.

### Statsmodels - Key Functions and Examples

| Category                  | Function or Example                                  | Description                                                                                                          |
| ------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Regression Models**     | `sm.OLS(y, X).fit()`                                 | Fit an Ordinary Least Squares (OLS) linear regression model.                                                         |
|                           | `sm.GLS(y, X).fit()`                                 | Fit a Generalized Least Squares (GLS) model.                                                                         |
| **Time Series Analysis**  | `sm.tsa.ARIMA(data, order=(1,1,1)).fit()`            | Fit an ARIMA model to time series data.                                                                              |
|                           | `sm.tsa.seasonal_decompose(x, model='additive')`     | Decompose a time series into its seasonal, trend, and residual components.                                           |
| **Statistical Tests**     | `sm.stats.ttest_ind(x1, x2)`                         | Calculate the T-test for the means of two independent samples of scores.                                             |
|                           | `sm.stats.ztest(x1, x2=None, value=0)`               | Perform a Z-test of the null hypothesis of the mean for one or two samples.                                          |
| **Diagnostic Plots**      | `sm.qqplot(data, line='s')`                          | Generate a Q-Q plot to compare the quantiles of a dataset to a theoretical distribution.                             |
|                           | `sm.qqplot(data, line='s', fit=True)`                | Generate a Q-Q plot with a line fitted to the quantiles to compare the dataset to a theoretical normal distribution. |
| **Nonparametric Methods** | `sm.nonparametric.KDEUnivariate(data).fit()`         | Fit a kernel density estimate for univariate data.                                                                   |
| **Discrete Models**       | `sm.Logit(y, X).fit()`                               | Fit a logistic regression model.                                                                                     |
|                           | `sm.Poisson(y, X).fit()`                             | Fit a Poisson regression model to count data.                                                                        |
| **Robust Linear Models**  | `sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()`     | Fit a robust linear model using Huber's T norm.                                                                      |
| **Factor Analysis**       | `sm.factor_analysis.Factor(endog, n_factor=2).fit()` | Perform factor analysis on multivariate data.                                                                        |
| **Survival Analysis**     | `sm.duration.SurvfuncRight(time, status).fit()`      | Estimate survival functions using Kaplan-Meier estimator for right-censored data.                                    |
| **Categorical Data**      | `sm.MNLogit(y, X).fit()`                             | Fit a multinomial logistic regression model for unordered categorical outcomes.                                      |
