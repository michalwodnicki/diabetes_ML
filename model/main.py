import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, mean_absolute_error
import pickle5 as pickle
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve


def create_model(data): 
  X = data.drop(['diabetes'], axis=1)
  y = data['diabetes']
  
  # scale the data
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  
  # split the data 80/10/10 train/validation/test
  X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.1, random_state=16, stratify=y
    )

  X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.1111, random_state=16, stratify=y_train_val
    )
  
  # train and fit the gbx model

  params = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.1,
    'random_state': 420
  }

  clf = GradientBoostingClassifier(**params)
  clf.fit(X_train, y_train)

  # test the gbx model
  y_pred = clf.predict(X_val)

  acc = accuracy_score(y_val, y_pred)

  prob_pos = clf.predict_proba(X_test)[:, 1]
  fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos, n_bins=10)
  uncalibrated_mae = mean_absolute_error(fraction_of_positives, mean_predicted_value)

  print(f"Validation Accuracy: {acc:.3f}")
  print(f"Validation Calibration Curve MAE: {uncalibrated_mae:.3f}")

  # train and fit calibration model

  calibrated_model = CalibratedClassifierCV(clf, cv='prefit', method ='isotonic')
  calibrated_model.fit(X_val, y_val)
  
  # test the calibration model
  y_pred_calibrated = calibrated_model.predict(X_test)

  acc_calibrated = accuracy_score(y_test, y_pred_calibrated)

  prob_pos_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
  fraction_of_positives_calibrated, mean_predicted_value_calibrated = calibration_curve(y_test, prob_pos_calibrated, n_bins=10)
  calibrated_mae = mean_absolute_error(fraction_of_positives_calibrated, mean_predicted_value_calibrated)

  print(f"Test Accuracy: {acc_calibrated:.3f}")
  print(f"Test Calibration Curve MAE Score: {calibrated_mae:.3f}")

  return calibrated_model, scaler


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
