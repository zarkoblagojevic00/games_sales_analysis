from eda.loader import GamesSalesLoader
from eda.visualizer import GamesSalesVisualizer
from eda.tester import GamesSalesTester

if __name__ == '__main__':
    loader = GamesSalesLoader(n_years=3)
    x, y = loader.load_X_y()

    tester = GamesSalesTester(x, y)
    tester.compare_results(show_pred=True)

    visualizer = GamesSalesVisualizer(loader.df, loader.genre_sales)
    visualizer.plot_all()