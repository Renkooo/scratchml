import cProfile
import pstats
from pstats import SortKey
import numpy as np
from sklearn.datasets import make_classification, make_regression
from scratchml.models.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

# profile_decision_tree.py

def profile_decision_tree():
    """Profile Decision Tree performance for both classification and regression"""
    
    # Generate classification sample data
    X_clf, y_clf = make_classification(
        n_samples=5000,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    
    # Generate regression sample data
    X_reg, y_reg = make_regression(
        n_samples=5000,
        n_features=20,
        random_state=42
    )

    # Test classification with different criteria
    criteria_clf = ['gini', 'entropy', 'log_loss']
    for criterion in criteria_clf:
        dt_clf = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=10,
            min_samples_split=5
        )
        dt_clf.fit(X_clf, y_clf)
        dt_clf.predict(X_clf)
        dt_clf.score(X_clf, y_clf)

    # Test regression with different criteria
    criteria_reg = ['squared_error', 'poisson']
    for criterion in criteria_reg:
        if criterion == 'poisson':
            # Poisson requires positive values
            y_reg_pos = np.abs(y_reg)
            dt_reg = DecisionTreeRegressor(
                criterion=criterion,
                max_depth=10,
                min_samples_split=5
            )
            dt_reg.fit(X_reg, y_reg_pos)
            dt_reg.predict(X_reg)
            dt_reg.score(X_reg, y_reg_pos)
        else:
            dt_reg = DecisionTreeRegressor(
                criterion=criterion,
                max_depth=10,
                min_samples_split=5
            )
            dt_reg.fit(X_reg, y_reg)
            dt_reg.predict(X_reg)
            dt_reg.score(X_reg, y_reg)

if __name__ == '__main__':
    # Profile trees
    profiler = cProfile.Profile()
    profiler.enable()
    profile_decision_tree()
    profiler.disable()
    
    # Save and print profiling stats
    stats = pstats.Stats(profiler)
    stats.sort_stats(SortKey.CUMULATIVE)
    stats.print_stats(20)
    stats.dump_stats('decision_tree_profile_afterChanges2.stats')