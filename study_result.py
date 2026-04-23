import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


data = {
    "hours_studied":[1,2,3,4,5,6,7,8],
    "exam_marks":[35,40,50,55,60,68,75,80]
}
df = pd.DataFrame(data)

print("Dataset: ")
print(df)


plt.scatter(df["hours_studied"],df["exam_marks"])
plt.xlabel("Hours studied")
plt.ylabel("Exam marks")
plt.title("Study hours vs Exam marks")
plt.show()



X=df[["hours_studied"]]
y=df["exam_marks"]
x_train,x_test,y_train,y_test= train_test_split(
    X,y,test_size=0.25,random_state=42
)
print("Traning Data")
print(x_train)
print("terst data")
print(x_test)



model= LinearRegression()
model.fit(x_train, y_train)




y_pred=model.predict(x_test)
print("Actual marks",y_test)
print("Predicted marks",y_pred)



mse=mean_squared_error(y_test,y_pred)
print("Mean squared error: ", mse)


new_hours=[[9]]
predicted_mark=model.predict(new_hours)
print(f"If a student studies 9 hours then his predicted mark:{predicted_mark[0]:.2f}")
plt.scatter(X,y , label="Actual data")
plt.plot(X,model.predict(X),label="Regression line")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Marks")
plt.title("Linear Regression")
plt.legend()
plt.show()

#JUST A COMMENTh
#JUST A COMMENT tseter
