from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'models/churn_model.pkl')
    return clf
