import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

sns.set(style="whitegrid")

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")

df.head()

#Data Overview
df.info()

df.describe()

df.isnull().sum()

#Data Cleaning
df = df.dropna()

df = df.drop(columns=["name"])

#Exploratory Data Analysis
sns.histplot(df["mpg"])
sns.scatterplot(
    data=df,
    x="weight",
    y="mpg"
)
sns.scatterplot(
    data=df,
    x="horsepower",
    y="mpg"
)
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")

plt.title("Distribution of MPG")
plt.xlabel("Miles Per Gallon")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="weight",
    y="mpg",
    hue="origin"
)

plt.title("Weight vs MPG")
plt.xlabel("Weight")
plt.ylabel("MPG")

plt.show()

plt.figure(figsize=(8,5))

sns.scatterplot(
    data=df,
    x="horsepower",
    y="mpg",
    hue="origin"
)

plt.title("Horsepower vs MPG")
plt.xlabel("Horsepower")
plt.ylabel("MPG")

plt.show()

#Feature Engineering
df = pd.get_dummies(
    df,
    columns=["origin"],
    prefix="origin"
)

#Train-Test Split
X = df[
[
'cylinders',
'displacement',
'horsepower',
'weight',
'acceleration',
'model_year',
'origin_europe',
'origin_japan',
'origin_usa'
]]

y = df["mpg"]

X_train,X_test,y_train,y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42
)

#Linear Regression
model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#Evaluation
r2 = r2_score(y_test,y_pred)

rmse = np.sqrt(mean_squared_error(y_test,y_pred))

print(r2)

print(rmse)

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred)
plt.xlabel("Actual MPG")
plt.ylabel("Predicted MPG")
plt.title("Actual vs Predicted MPG")
plt.show()

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="mpg",
    bins=20,
    color="skyblue"
)



#Key Findings
#- Vehicle weight has a strong negative impact on fuel efficiency.
#- Horsepower is negatively correlated with MPG.
#- Newer vehicles generally have higher fuel efficiency.
#- The Linear Regression model achieved:
#    • R² = 0.79
#    • RMSE = 3.26

#Conclusion
#The Linear Regression model successfully predicts vehicle fuel efficiency with good performance.
#The model explains approximately 79% of the variation in MPG.
#Vehicle weight, horsepower, and production year are among the most influential predictors.
