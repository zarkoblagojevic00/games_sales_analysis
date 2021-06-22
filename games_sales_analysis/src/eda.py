import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# <editor-fold desc="Visualization functions">
def plot_genre_sales_to_count_comparison(genre_sales):
    marketplace_count = len(sales_cols)
    fig, axes = plt.subplots(figsize=(10, 32), nrows=marketplace_count, ncols=1, sharex=False)

    for ax, region_sales in zip(axes, sales_cols):
        sns.scatterplot(ax=ax, data=genre_sales, x='Count', y=region_sales)
        ax.set_title(f'Comparison of Games Sales and Count for {region_sales}', fontsize=18)
        ax.set_xlabel('Count of games made', fontsize=16)
        ax.set_ylabel(f'{region_sales} (in millions)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.show()


def plot_genre_sales_comparison_for_year(genre_sales_per_year_comparison: pd.DataFrame, year: int):
    genre_sales_for_one_year_comparison = genre_sales_per_year_comparison[('Global_Sales', year)]

    plt.rcParams['figure.figsize'] = (8, 6)
    sns.barplot(x=genre_sales_for_one_year_comparison.values, y=genre_sales_for_one_year_comparison.index, orient='h')
    plt.title(f'Genre global sales comparison for year {year}')
    plt.xlabel('Global sales (in millions)')

    plt.show()


def plot_genre_sales_history(genre_sales_history: pd.DataFrame, genre: str):
    one_genre_sales_history = genre_sales_history[('Global_Sales', genre)]

    plt.rcParams['figure.figsize'] = (8, 6)
    sns.barplot(x=one_genre_sales_history.values, y=one_genre_sales_history.index, orient='h')
    plt.title(f'Global sales history for genre {genre}')
    plt.xlabel('Global sales (in millions)')

    plt.show()


def plot_sales_correlation(df_sales):
    correlation = df_sales.corr()
    sns.heatmap(correlation, annot=True)
    plt.show()


def plot_sales_rsquared(df_sales):
    correlation = df_sales.corr()
    r_squared = np.power(correlation, 2)
    sns.heatmap(r_squared, annot=True)
    plt.show()


def plot_residual_NA_Sales(genre_sales):
    sns.residplot(data=genre_sales, x='NA_Sales', y='Global_Sales')
    plt.title('Residual plot of NA_Sales vs Global_sales')
    plt.ylabel('Residuals')
    plt.show()
# </editor-fold>

# <editor-fold desc="Data IO and df creation">
path = os.path.join("..", "data", "vgsales.csv")
df = pd.read_csv(path)
print(df.shape)
df.dropna(axis=0, how='any', inplace=True)
print(df.shape)
sales_cols = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
index_cols = ['Year', 'Genre']
aggfunc_dict = dict.fromkeys(sales_cols, lambda x: np.sum(x))
aggfunc_dict['Rank'] = pd.Series.nunique

genre_sales = pd.pivot_table(data=df,
                             index=index_cols,
                             aggfunc=aggfunc_dict,
                             dropna=False,
                             fill_value=0)
genre_sales.rename(columns={'Rank': 'Count'}, inplace=True)

genre_sales_per_year_comparison = genre_sales.unstack(0)
genre_sales_history = genre_sales.unstack(-1)
# </editor-fold>

# <editor-fold desc="Linear regression">
attributes = genre_sales[['NA_Sales', 'EU_Sales', 'Count']]
target = genre_sales['Global_Sales']

xtrain, xtest, ytrain, ytest=train_test_split(attributes, target, test_size=.3, random_state=1)
lr_model = LinearRegression()
lr_model.fit(xtrain, ytrain)
ypred=lr_model.predict(xtest)
n = len(xtest)
p = xtest.shape[1]
r2_value = r2_score(ytest, ypred)

result_df = pd.DataFrame({'Test': ytest, 'Predict': ypred})

adjusted_r2_score = 1 - (((1-r2_value)*(n-1)) / (n-p-1))
print("r2_score for Linear Reg model : ", r2_score(ytest, ypred))
print("adjusted_r2_score Value       : ", adjusted_r2_score)
print("MSE for Linear Regression     : ", mean_squared_error(ytest, ypred))
# </editor-fold>

plot_genre_sales_comparison_for_year(genre_sales_per_year_comparison, year=2014)
plot_genre_sales_history(genre_sales_history, 'Sports')
plot_genre_sales_to_count_comparison(genre_sales)
plot_sales_correlation(df[sales_cols])
plot_sales_rsquared(genre_sales)
plot_residual_NA_Sales(genre_sales)







