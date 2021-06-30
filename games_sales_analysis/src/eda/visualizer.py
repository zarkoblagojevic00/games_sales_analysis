import matplotlib.pyplot as plt
import seaborn as sns


class GamesSalesVisualizer:

    def __init__(self, df, genre_sales):
        self.df = df
        self.genre_sales = genre_sales
        self.sales_cols_names = [name for name in df.columns if '_Sales' in name]

    def plot_all(self):
        self.plot_genre_sales_comparison_for_year(year=2014)
        self.plot_genre_sales_history('Sports')
        self.plot_genre_sales_to_count_comparison()
        self.plot_sales_correlation()
        self.plot_sales_rsquared()
        self.plot_residual_NA_Sales()

    def plot_genre_sales_to_count_comparison(self):
        marketplace_count = len(self.sales_cols_names)
        fig, axes = plt.subplots(figsize=(10, 32), nrows=marketplace_count, ncols=1, sharex=False)

        for ax, region_sales in zip(axes, self.sales_cols_names):
            sns.scatterplot(ax=ax, data=self.genre_sales, x='Count', y=region_sales)
            ax.set_title(f'Comparison of Games Sales and Count for {region_sales}', fontsize=18)
            ax.set_xlabel('Count of games made', fontsize=16)
            ax.set_ylabel(f'{region_sales} (in millions)', fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.show()

    def plot_genre_sales_comparison_for_year(self, year: int):
        genre_sales_per_year_comparison = self.genre_sales.unstack(0)
        genre_sales_for_one_year_comparison = genre_sales_per_year_comparison[('Global_Sales', year)]

        plt.rcParams['figure.figsize'] = (8, 6)
        sns.barplot(x=genre_sales_for_one_year_comparison.values, y=genre_sales_for_one_year_comparison.index,
                    orient='h')
        plt.title(f'Genre global sales comparison for year {year}')
        plt.xlabel('Global sales (in millions)')

        plt.show()

    def plot_genre_sales_history(self, genre: str):
        genre_sales_history = self.genre_sales.unstack()
        one_genre_sales_history = genre_sales_history[('Global_Sales', genre)]

        plt.rcParams['figure.figsize'] = (8, 6)
        sns.barplot(x=one_genre_sales_history.values, y=one_genre_sales_history.index, orient='h')
        plt.title(f'Global sales history for genre {genre}')
        plt.xlabel('Global sales (in millions)')

        plt.show()

    def plot_sales_correlation(self):
        correlation = self.df[self.sales_cols_names].corr()
        sns.heatmap(correlation, annot=True)
        plt.show()

    def plot_sales_rsquared(self):
        correlation = self.genre_sales.corr()
        r_squared = correlation ** 2
        sns.heatmap(r_squared, annot=True)
        plt.show()

    def plot_residual_NA_Sales(self):
        sns.residplot(data=self.genre_sales, x='NA_Sales', y='Global_Sales')
        plt.title('Residual plot of NA_Sales vs Global_sales')
        plt.ylabel('Residuals')
        plt.show()



