import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.metrics import confusion_matrix, r2_score, mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, precision_recall_curve
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/Train.txt",sep=",",names=["duration","protocoltype","service","flag","srcbytes","dstbytes","land", "wrongfragment","urgent","hot","numfailedlogins","loggedin", "numcompromised","rootshell","suattempted","numroot","numfilecreations", "numshells","numaccessfiles","numoutboundcmds","ishostlogin",
"isguestlogin","count","srvcount","serrorrate", "srvserrorrate",
"rerrorrate","srvrerrorrate","samesrvrate", "diffsrvrate", "srvdiffhostrate","dsthostcount","dsthostsrvcount","dsthostsamesrvrate", "dsthostdiffsrvrate","dsthostsamesrcportrate",
"dsthostsrvdiffhostrate","dsthostserrorrate","dsthostsrvserrorrate",
"dsthostrerrorrate","dsthostsrvrerrorrate","attack", "lastflag"])
df.head()
df.shape

df.describe()

"""As we can see 'land', 'urgent', 'numfailedlogins', 'numoutboundcmds' have mostly zero values so we can drop these columns."""

df.drop(['land','urgent','numfailedlogins','numoutboundcmds'],axis=1,inplace=True)

df.isna().sum()

df.select_dtypes(exclude=[np.number])

"""As we are focussing on Binomial Classification for this dataset, we can make all other classes other than normal as 'attack'"""

df['attack'].loc[df['attack']!='normal']='attack'

le=LabelEncoder()

df['protocoltype']=le.fit_transform(df['protocoltype'])
df['service']=le.fit_transform(df['service'])
df['flag']=le.fit_transform(df['flag'])
df['attack']=le.fit_transform(df['attack'])

plt.figure(figsize=(20,15))
sns.heatmap(df.corr())

X=df.drop(['attack'],axis=1)
y=df['attack']

sns.countplot(df['attack'])

print("Class distribution: {}".format(Counter(y)))

scaler = StandardScaler()
scaler.fit(X)
X_transformed = scaler.transform(X)

"""**Using Logistic Regression**"""

lr=LogisticRegression() # creates an instance of the Logistic Regression model.
lr.fit(X_transformed,y) #  trains the Logistic Regression model on the training data. X_transformed represents the feature matrix (input variables), and y represents the target variable (labels or classes).
lr_pred=lr.predict(X_transformed) # generates predictions for the training set based on the trained Logistic Regression model.

lr_df=pd.DataFrame()
lr_df['actual']=y
lr_df['pred']=lr_pred

lr_df.head()

print(accuracy_score(y, lr_pred))

confusion_matrix(y, lr_pred)

print(classification_report(y, lr_pred))

"""**Using Random Forest Classifier**"""

rf=RandomForestClassifier()
rf.fit(X_transformed,y)
rf_pred=rf.predict(X_transformed)

rf_df=pd.DataFrame()
rf_df['actual']=y
rf_df['pred']=rf_pred
rf_df.head()

print(accuracy_score(y, rf_pred))

confusion_matrix(y, rf_pred)

print(classification_report(y, rf_pred))

"""**Using SVM**"""

svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_transformed, y)
svm_pred=svm_model.predict(X_transformed)

print(accuracy_score(y, svm_pred))
print(classification_report(y, svm_pred))

"""**Using Random Forest Classifier Model on test data:**"""

test_df = pd.read_csv("data/Test.txt",sep=",",names=["duration","protocoltype","service","flag","srcbytes","dstbytes","land", "wrongfragment","urgent","hot","numfailedlogins","loggedin", "numcompromised","rootshell","suattempted","numroot","numfilecreations", "numshells","numaccessfiles","numoutboundcmds","ishostlogin",
"isguestlogin","count","srvcount","serrorrate", "srvserrorrate",
"rerrorrate","srvrerrorrate","samesrvrate", "diffsrvrate", "srvdiffhostrate","dsthostcount","dsthostsrvcount","dsthostsamesrvrate", "dsthostdiffsrvrate","dsthostsamesrcportrate",
"dsthostsrvdiffhostrate","dsthostserrorrate","dsthostsrvserrorrate",
"dsthostrerrorrate","dsthostsrvrerrorrate","attack", "lastflag"])
test_df.head()

test_df.select_dtypes(exclude=[np.number])

test_df['attack'].loc[test_df['attack']!='normal']='attack'

test_df['protocoltype']=le.fit_transform(test_df['protocoltype'])
test_df['service']=le.fit_transform(test_df['service'])
test_df['flag']=le.fit_transform(test_df['flag'])
test_df['attack']=le.fit_transform(test_df['attack'])

test_df.drop(['land','urgent','numfailedlogins','numoutboundcmds'],axis=1,inplace=True)

X_test=test_df.drop(['attack'],axis=1)
y_test=test_df['attack']

sns.countplot(test_df['attack'])

X_test_transformed = scaler.transform(X_test)

test_pred=rf.predict(X_test_transformed)

rf_test_df=pd.DataFrame()
rf_test_df['actual']=y_test
rf_test_df['pred']=test_pred

rf_test_df.head()

print(accuracy_score(y_test, test_pred))

target_names=["attack","normal"]

print(classification_report(y_test, test_pred,target_names=target_names))

confusion_matrix(y_test, test_pred)

"""**Logistic Regression**"""

test_pred=lr.predict(X_test_transformed)
print(classification_report(y_test, test_pred,target_names=target_names))

lr_pred=lr.score(X_test_transformed, y_test)
print(lr_pred)

"""**SVM**"""

test_pred=svm_model.predict(X_test_transformed)
print(classification_report(y_test, test_pred,target_names=target_names))

accuracy_score = svm_model.score(X_test_transformed, y_test)
print(accuracy_score)

from sklearn.metrics import roc_curve, auc

y_prob = svm_model.predict(X_test_transformed)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_transformed.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    X_transformed, 
    y,
    epochs=20,
    batch_size=32,
    validation_split=0.2)

training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']
epochs = range(1, len(training_accuracy) + 1)

# Plot training and validation accuracy
plt.plot(epochs, training_accuracy, 'b', label='Training accuracy')
plt.plot(epochs, validation_accuracy, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss, test_acc = model.evaluate(X_test_transformed, y_test)
print(f'Test accuracy: {test_acc}')

predictions = model.predict(X_test_transformed)
predictions = (predictions > 0.5)

model.summary()

y_pred = (predictions > 0.5).astype(int)

report = classification_report(y_test, y_pred)

print("Classification Report:")
print(report)




import plotly.express as px

# Define pastel colors
pastel_colors = ['#FFD1DC', '#FFA07A', '#B0E0E6', '#98FB98']

# Create DataFrame
data = {
    "Model": ["Random Forest", "SVC", "Logistic Regression", "Neural Networks"],
    "Accuracy": [82, 86, 84, 87],
    "Color": pastel_colors
}

df = pd.DataFrame(data)

# Create bar plot
fig = px.bar(df, x="Model", y="Accuracy", text="Accuracy",
             color="Color", color_discrete_map="identity")

fig.update_traces(texttemplate='%{text}%', textposition='outside')

fig.update_layout(title="Accuracy of Different Machine Learning Models",
                  xaxis_title="Model",
                  yaxis_title="Accuracy (%)")

fig.show()

