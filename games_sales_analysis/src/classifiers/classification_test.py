from sklearn import datasets
from src.eda.tester import ClassificationTester

if __name__ == '__main__':
    iris = datasets.load_iris()
    iris_tester = ClassificationTester(iris.data, iris.target)
    iris_tester.compare_results()

    wine = datasets.load_wine()
    wine_tester = ClassificationTester(wine.data, wine.target)
    wine_tester.compare_results()
