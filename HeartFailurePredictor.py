import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_clinical_records_dataset.csv")

df = load_data()

# Title and Description
st.title("Heart Failure Prediction - ML Model Comparison")
st.markdown("This application helps us understand machine learning models and their effectiveness in predicting heart failure.")

# Show dataset preview
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Feature Importance Analysis
st.header("Feature Importance Analysis")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
model = ExtraTreesClassifier()
model.fit(X, y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)

st.subheader("Top 10 Most Important Features")
fig, ax = plt.subplots()
feat_importances.nlargest(10).plot(kind='barh', ax=ax)
ax.set_title("Top 10 Feature Importances")
st.pyplot(fig)

# Scatter plot of two most important features
st.subheader("Relationship Between Features and Target")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='time', y='ejection_fraction', hue='DEATH_EVENT', ax=ax)
ax.set_title("Time vs Ejection Fraction with Death Event")
st.pyplot(fig)

# Train-Test Split
st.header("Model Training & Accuracy Comparison")
st.markdown("The models are trained using the two most important features: `time` and `ejection_fraction`.")
X_selected = df[['time', 'ejection_fraction']].values
y = df['DEATH_EVENT'].values
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.20, random_state=2)

# Define models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVC': SVC(),
    'KNeighbors': KNeighborsClassifier(n_neighbors=6),
    'Decision Tree': DecisionTreeClassifier(max_leaf_nodes=3, random_state=0, criterion='entropy'),
    'Random Forest': RandomForestClassifier(max_features=0.5, max_depth=15, random_state=1)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Show accuracy results
st.subheader("Model Accuracy Comparison")
fig, ax = plt.subplots()
ax.bar(results.keys(), results.values())
ax.set_title("Model Accuracy Comparison")
ax.set_xlabel("Models")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.8, 1)
plt.xticks(rotation=45, ha='right')
st.pyplot(fig)

# Display accuracy scores
for name, accuracy in results.items():
    st.write(f"**{name}:** {accuracy * 100:.2f}%")

# Prediction Section
st.header("Make a Prediction")
st.markdown("Enter values for `time` and `ejection_fraction` to predict the likelihood of heart failure.")

time = st.number_input("Time (Follow-up period in days)", min_value=0, max_value=500, value=100)
ejection_fraction = st.number_input("Ejection Fraction (%)", min_value=10, max_value=80, value=40)

selected_model = st.selectbox("Choose a Model", list(models.keys()))

if st.button("Predict"):
    model = models[selected_model]
    prediction = model.predict([[time, ejection_fraction]])
    result = "Likely to survive" if prediction[0] == 0 else "High risk of death"
    st.subheader(f"Prediction: {result}")
