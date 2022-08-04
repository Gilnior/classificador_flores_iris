import pandas as pd
from classificador_flores_iris.classificador import *

def main() -> None:
    iris = pd.read_csv("dataset/iris.csv")

    x = iris.drop("species", axis=1).values
    y = iris.species.map({'setosa': 1, 'versicolor': 2, 'virginica': 3}).values

    result = classificador_flores_iris(x, y, x)

    print(np.sum(y == result)/len(result))  # silly score

if __name__ == "__main__":
    main()
