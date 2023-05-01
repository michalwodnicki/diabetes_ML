import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, f1_score
import pickle5 as pickle


def create_model(data):
    X = data.drop(["diabetes"], axis=1)
    y = data["diabetes"]

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=420
    )

    # train the model

    params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.1,
        "random_state": 420,
    }

    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)

    # test model

    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    print("F1 score:", f1)

    print("Accuracy of our model: ", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    return model, scaler


def get_clean_data():
    # parent = Path(__name__).parent
    # filename = os.path.join(parent, "data", "data.csv")
    data_path = "./data/data.csv"
    data = pd.read_csv(data_path)

    data["gender"] = data["gender"].map({"Male": 0, "Female": 1})

    smoker_converted = pd.get_dummies(data["smoking_history"], drop_first=True)
    data = pd.concat([data, smoker_converted], axis=1)
    data.drop("smoking_history", axis=1, inplace=True)

    data["never"] = data.apply(lambda row: row["never"] + row["ever"], axis=1)
    data.drop(["ever"], axis=1, inplace=True)
    
    data = data.rename(columns={'not current': 'not_current'})
    data["former"] = data["not_current"] + data["former"]
    data.drop(['not_current'], axis=1, inplace=True)

    data = data.dropna()

    contam = 15 / 99982
    outliers = False
    # Removing outliers with isolation forest
    if outliers:
        clf = IsolationForest(contamination=contam, random_state=42)
        clf.fit(data)
        is_outlier = clf.predict(data)
        data = data[is_outlier == 1]

    #print(len(data))

    return data


def main():
    data = get_clean_data()

    model, scaler = create_model(data)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
