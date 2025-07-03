from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def tune_model(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8, 10],
        'min_samples_split': [2, 5],
    }
    grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=3, scoring='accuracy')
    grid.fit(X_train, y_train)
    return grid.best_estimator_
