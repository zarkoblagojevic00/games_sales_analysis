import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


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
                             fill_value=0).rename(columns={'Rank': 'Count'})

genre_sales_per_year_comparison = genre_sales.unstack(0)
genre_sales_history = genre_sales.unstack(-1)


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


plot_genre_sales_comparison_for_year(genre_sales_per_year_comparison, year=2010)
plot_genre_sales_history(genre_sales_history, 'Strategy')
plot_genre_sales_to_count_comparison(genre_sales)
