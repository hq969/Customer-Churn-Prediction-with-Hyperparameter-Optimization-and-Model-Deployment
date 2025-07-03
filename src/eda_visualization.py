import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def eda(df):
    sns.countplot(data=df, x='Exited')
    plt.title("Churn Distribution")
    plt.show()

    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation")
    plt.show()
