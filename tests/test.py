from auto_tabnet import AutoTabnetClassifier
import pandas as pd

if __name__ == "__main__":

    df = pd.read_csv('./winetrain.csv')
    df2 = pd.read_csv('./winetest.csv')

    X = df.drop("Output", axis=1)   
    y = df.Output
  
    X_test = df2

    clf = AutoTabnetClassifier(X, y, X_test)

    results = clf.predict()
    


