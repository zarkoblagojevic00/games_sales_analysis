from collections import Counter

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

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

# <editor-fold desc="Data IO and df creation"


path = os.path.join("..", "data", "vgsales.csv")
df = pd.read_csv(path)
print(df.shape)
df.dropna(axis=0, how='any', inplace=True)
df.drop(df[df['Year'] > 2016].index, inplace=True)
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


def get_most_popular_genre(year_row):
    genre_global_sales_for_year = year_row.loc['Global_Sales']
    most_popular_genre = genre_global_sales_for_year.idxmax()
    return most_popular_genre


def get_last_if_multiple_modes_without_sort(x: pd.Series):
    if len(x) == 0:
        return None

    counts = Counter(x)
    max_count = max(counts.values())

    for elem, count in counts.items():
        if count == max_count:
            last_mode = elem

    return last_mode


def get_most_popular_genres_in_last_n_years(most_pop_genres, n_years):
    years = most_pop_genres.index
    min_year = years.values.min()
    most_pop_genres_in_last_n_years = pd.Series(index=years, dtype=str)

    for idx, year in enumerate(years):
        begin_year = min_year if idx < n_years else year - n_years
        most_pop_genres_in_time_span = most_pop_genres[begin_year: year - 1]
        most_pop_genres_in_last_n_years[year] = get_last_if_multiple_modes_without_sort(most_pop_genres_in_time_span)

    return most_pop_genres_in_last_n_years


def get_most_popular_genres_n_years_ago(most_pop_genres, n_years_ago):
    return most_pop_genres.shift(periods=n_years_ago)


most_popular_genres = genre_sales_history.apply(func=get_most_popular_genre, axis=1)
genre_sales_history['Most_Pop_Genre'] = most_popular_genres

n_years = 3

most_popular_genres_in_last_n_years = get_most_popular_genres_in_last_n_years(most_popular_genres, n_years)
genre_sales_history[f'Most_Pop_Genre_Last_{n_years}_Years'] = most_popular_genres_in_last_n_years

genre_sales_history['Most_Pop_Genre'] = most_popular_genres.factorize()[0]
genre_sales_history[f'Most_Pop_Genre_Last_{n_years}_Years'] = most_popular_genres_in_last_n_years.factorize()[0]

genre_sales_history[f'Most_Pop_Genre_{n_years}_Years_Ago'] = get_most_popular_genres_n_years_ago(most_popular_genres, n_years)
genre_sales_history[f'Most_Pop_Genre_{n_years - 1}_Years_Ago'] = get_most_popular_genres_n_years_ago(most_popular_genres, n_years - 1)
genre_sales_history[f'Most_Pop_Genre_{n_years - 2}_Years_Ago'] = get_most_popular_genres_n_years_ago(most_popular_genres, n_years - 2)

cols_to_encode = ['Most_Pop_Genre',
                  f'Most_Pop_Genre_Last_{n_years}_Years',
                  f'Most_Pop_Genre_{n_years}_Years_Ago',
                  f'Most_Pop_Genre_{n_years - 1}_Years_Ago',
                  f'Most_Pop_Genre_{n_years - 2}_Years_Ago']

genre_sales_history[cols_to_encode] = genre_sales_history[cols_to_encode].transform(func=lambda x: x.factorize()[0])
# </editor-fold>

# <editor-fold desc="Train_test splitting">
attr_cols = ['NA_Sales', 'Count', f'Most_Pop_Genre_Last_{n_years}_Years']

attr_cols = [ f'Most_Pop_Genre_Last_{n_years}_Years',
              f'Most_Pop_Genre_{n_years}_Years_Ago',
              f'Most_Pop_Genre_{n_years - 1}_Years_Ago',
              f'Most_Pop_Genre_{n_years - 2}_Years_Ago',
              ]
attr_cols = ['NA_Sales']

clf_attr = genre_sales_history[attr_cols]
clf_target = genre_sales_history['Most_Pop_Genre']
x_train, x_test, y_train, y_test = train_test_split(clf_attr, clf_target, test_size=.3, random_state=0)

# </editor-fold>

# <editor-fold desc="Logistic regression">
clf = LogisticRegression(random_state=0, max_iter=500).fit(x_train, y_train)
pred = clf.predict(x_test)
results = pd.DataFrame({'True': y_test, 'Predicted': pred})
print('\n----- Logistic Regression results -----')
print(results)
print(40 * '-')
score = clf.score(x_test, y_test)
print(f'Score : {score}')
# </editor-fold>

# <editor-fold desc="SVM Classification">
clf = svm.SVC(kernel='linear').fit(x_train, y_train)
pred = clf.predict(x_test)
results = pd.DataFrame({'True': y_test, 'Predicted': pred})
print('\n----- SVM Classification results -----')
print(results)
print(40 * '-')
score = clf.score(x_test, y_test)
print(f'Score : {score}')
# </editor-fold>

# <editor-fold desc="RandomForest Classification">
clf = RandomForestClassifier(random_state=1, bootstrap=True, max_features="sqrt").fit(x_train, y_train)
pred = clf.predict(x_test)
results = pd.DataFrame({'True': y_test, 'Predicted': pred})
print('\n----- RandomForest Classification results -----')
print(results)
print(40 * '-')
score = clf.score(x_test, y_test)
print(f'Score : {score}')
# </editor-fold>

# <editor-fold desc="Linear regression">
normalized_genre_sales = (genre_sales-genre_sales.mean())/genre_sales.std()
attributes = normalized_genre_sales[['NA_Sales', 'Count']]
target = normalized_genre_sales['Global_Sales']

xtrain, xtest, ytrain, ytest=train_test_split(attributes, target, test_size=.3, random_state=1)
lr_model = LinearRegression()
lr_model.fit(xtrain, ytrain)
ypred=lr_model.predict(xtest)
result_df = pd.DataFrame({'Test': ytest, 'Predict': ypred})

r2_value = r2_score(ytest, ypred)
n = len(xtest)
p = xtest.shape[1]

adjusted_r2_score = 1 - (((1-r2_value)*(n-1)) / (n-p-1))

print("r2_score for Linear Reg model : ", r2_value)
print("adjusted_r2_score Value       : ", adjusted_r2_score)
print("MSE for Linear Regression     : ", mean_squared_error(ytest, ypred))

# </editor-fold>


plot_genre_sales_comparison_for_year(genre_sales_per_year_comparison, year=2014)
plot_genre_sales_history(genre_sales_history, 'Sports')
plot_genre_sales_to_count_comparison(genre_sales)
plot_sales_correlation(df[sales_cols])
plot_sales_rsquared(genre_sales)
plot_residual_NA_Sales(genre_sales)






