# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

# Step 1: Load dataset
file_path = './emissions.csv'  # Adjust path if necessary
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# Preprocess the dataset
# Encoding categorical variables
data_encoded = data.copy()
label_encoders = {}
for column in ['state-name', 'sector-name', 'fuel-name']:
    le = LabelEncoder()
    data_encoded[column] = le.fit_transform(data_encoded[column])
    label_encoders[column] = le

# Binning the target variable into discrete classes
k_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
data_encoded['value'] = k_bins.fit_transform(data_encoded[['value']]).astype(int)

# Features and target variable
X = data_encoded.drop('value', axis=1)  # Features
y = data_encoded['value']  # Target variable

# Optional: Scale the data for models that require normalization
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Apply Ensemble Techniques

# 1. Voting Classifier
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = SVC(probability=True)

voting_clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('svc', clf3)], voting='soft')
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)

# 2. Bootstrap Aggregation (Bagging)
bagging_clf = BaggingClassifier(LogisticRegression(), n_estimators=50, random_state=42)
bagging_clf.fit(X_train, y_train)
bagging_pred = bagging_clf.predict(X_test)

# 3. Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)

# 4. Boosting (Gradient Boosting)
boosting_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
boosting_clf.fit(X_train, y_train)
boosting_pred = boosting_clf.predict(X_test)

# 5. Stacking Classifier
estimators = [('rf', RandomForestClassifier()), ('svc', SVC(probability=True))]
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stacking_clf.fit(X_train, y_train)
stacking_pred = stacking_clf.predict(X_test)

# Step 3: Evaluate Models
methods = ['Voting', 'Bagging', 'Random Forest', 'Boosting', 'Stacking']
predictions = [voting_pred, bagging_pred, rf_pred, boosting_pred, stacking_pred]

results = []
for i, pred in enumerate(predictions):
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average='weighted', zero_division=0)  # Handle zero division
    recall = recall_score(y_test, pred, average='weighted', zero_division=0)        # Handle zero division
    f1 = f1_score(y_test, pred, average='weighted')
    results.append([methods[i], accuracy, precision, recall, f1])

# Create a DataFrame to compare results
comparison_df = pd.DataFrame(results, columns=['Method', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
print(comparison_df)
