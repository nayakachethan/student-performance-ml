import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = {
    'Attendance': [90, 75, 60, 85, 50, 95, 40],
    'StudyHours': [4, 3, 1, 3.5, 1, 5, 0.5],
    'PreviousMarks': [80, 65, 45, 70, 40, 90, 35],
    'Result': [1, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[['Attendance', 'StudyHours', 'PreviousMarks']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

new_student = pd.DataFrame(
    [[78, 2.5, 65]],
    columns=['Attendance', 'StudyHours', 'PreviousMarks']
)

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")
