from eda.loader import GamesSalesLoader
from eda.visualizer import GamesSalesVisualizer
from eda.tester import ClassificationTester

if __name__ == '__main__':
    loader = GamesSalesLoader(n_years=3)
    x, y = loader.load_X_y()

    tester = ClassificationTester(x, y, tts_strat='leave_one_out')
    tester.compare_results(show_pred=True)

    visualizer = GamesSalesVisualizer(loader.df, loader.genre_sales)
    visualizer.plot_all()