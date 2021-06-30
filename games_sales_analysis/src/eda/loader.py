import pandas as pd
import os
from collections import Counter


class GamesSalesLoader:

    def __init__(self, n_years):
        self.df = None
        self.genre_sales = None
        self.genre_sales_history = None
        self.n_years = n_years
        self.__init_data_sets()

    def load_X_y(self):
        feat_cols = ['NA_Sales']
        feat_cols.append('Count')

        #feat_cols.append(f'Most_Pop_Genre_Last_{self.n_years}_Years')
        feat_cols.extend(self.__create_col_names_to_encode())

        X = self.genre_sales_history[feat_cols]
        y = self.genre_sales_history['Most_Pop_Genre']
        return X, y

    def __init_data_sets(self):
        self.__clean_load()
        self.__create_genre_sales()
        self.__create_genre_sales_history()

    def __clean_load(self):
        path = os.path.join("..", "data", "vgsales.csv")
        self.df = pd.read_csv(path)
        print(self.df.shape)
        self.df.dropna(axis=0, how='any', inplace=True)
        self.df.drop(self.df[self.df['Year'] > 2016].index, inplace=True)
        print(self.df.shape)

    def __create_genre_sales(self):
        sales_cols = ['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        index_cols = ['Year', 'Genre']
        aggfunc_dict = dict.fromkeys(sales_cols, lambda x: sum(x))
        aggfunc_dict['Rank'] = pd.Series.nunique

        self.genre_sales = pd.pivot_table(data=self.df,
                                          index=index_cols,
                                          aggfunc=aggfunc_dict,
                                          dropna=False,
                                          fill_value=0)
        self.genre_sales.rename(columns={'Rank': 'Count'}, inplace=True)

    def __create_genre_sales_history(self):
        self.genre_sales_history = self.genre_sales.unstack()
        self.__add_features_to_genre_sales_history()

    def get_most_popular_genre(self, year_row):
        genre_global_sales_for_year = year_row.loc['Global_Sales']
        most_popular_genre = genre_global_sales_for_year.idxmax()
        return most_popular_genre

    def get_last_if_multiple_modes_without_sort(self, x: pd.Series):
        if len(x) == 0:
            return None

        counts = Counter(x)
        max_count = max(counts.values())

        for elem, count in counts.items():
            if count == max_count:
                last_mode = elem

        return last_mode

    def get_most_popular_genres_in_last_n_years(self, most_pop_genres, n_years):
        years = most_pop_genres.index
        min_year = years.values.min()
        most_pop_genres_in_last_n_years = pd.Series(index=years, dtype=str)

        for idx, year in enumerate(years):
            begin_year = min_year if idx < n_years else year - n_years
            most_pop_genres_in_time_span = most_pop_genres[begin_year: year - 1]
            most_pop_genres_in_last_n_years[year] = self.get_last_if_multiple_modes_without_sort(
                most_pop_genres_in_time_span)

        return most_pop_genres_in_last_n_years

    def get_most_popular_genres_n_years_ago(self, most_pop_genres, n_years_ago):
        return most_pop_genres.shift(periods=n_years_ago)

    def __add_features_to_genre_sales_history(self):
        most_popular_genres = self.__create_target_feature()

        self.__create_most_popular_genres_in_last_n_years(most_popular_genres)

        self.__create_most_popular_genres_for_each_year_in_last_n_years(most_popular_genres)

        self.__encode_genre_columns()

    def __create_target_feature(self):
        most_popular_genres = self.genre_sales_history.apply(func=self.get_most_popular_genre, axis=1)
        self.genre_sales_history['Most_Pop_Genre'] = most_popular_genres
        return most_popular_genres

    def __create_most_popular_genres_in_last_n_years(self, most_popular_genres):
        most_popular_genres_in_last_n_years = self.get_most_popular_genres_in_last_n_years(most_popular_genres, self.n_years)
        self.genre_sales_history[f'Most_Pop_Genre_Last_{self.n_years}_Years'] = most_popular_genres_in_last_n_years

    def __create_most_popular_genres_for_each_year_in_last_n_years(self, most_popular_genres):
        for year in range(self.n_years):
            real_year = self.n_years - year
            self.genre_sales_history[
                f'Most_Pop_Genre_{real_year}_Years_Ago'] = self.get_most_popular_genres_n_years_ago(
                most_popular_genres,
                real_year)

    def __encode_genre_columns(self):
        cols_to_encode = self.__create_col_names_to_encode()
        self.genre_sales_history[cols_to_encode] = self.genre_sales_history[cols_to_encode].transform(
            func=lambda x: x.factorize()[0])

    def __create_col_names_to_encode(self):
        col_names = [f'Most_Pop_Genre_Last_{self.n_years}_Years']
        for year in range(self.n_years):
            col_names.append(f'Most_Pop_Genre_{self.n_years - year}_Years_Ago')

        return col_names
