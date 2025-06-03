##  Breast Cancer Diagnosis using K-Nearest Neighbors (KNN)

This project applies **K-Nearest Neighbors (KNN)** classification to predict breast cancer diagnosis (Malignant or Benign) using the `texture_mean` feature from a medical dataset. It includes preprocessing, model training, hyperparameter tuning, and visualization of the decision boundary.

---

###  Dataset

* **Name**: Cancer\_Data.csv
* **Features used**:

  * `texture_mean` (input)
  * `diagnosis` (output — 'M' for Malignant, 'B' for Benign)

---

###  Steps Performed

#### 1. **Import Required Libraries**

* `pandas`, `numpy` for data manipulation
* `matplotlib`, `seaborn` for visualization
* `sklearn` for machine learning models and tools

#### 2. **Data Loading and Inspection**

* Loaded CSV dataset using `pandas`
* Inspected structure using `info()`, `describe()`, and `isnull()`

#### 3. **Preprocessing**

* Removed irrelevant feature `id`
* Selected `texture_mean` as input
* Applied **StandardScaler** to normalize input features

#### 4. **Train-Test Split**

* Used `train_test_split` to split data (70% training, 30% testing)

#### 5. **Model Training**

* Trained a baseline **KNeighborsClassifier** using default `k`
* Evaluated using:

  * Accuracy Score
  * Confusion Matrix
  * Classification Report

#### 6. **Hyperparameter Tuning**

* Manually tried `n_neighbors=7`
* Used **GridSearchCV** to automatically tune:

  * `n_neighbors`: \[1, 3, 5, 7]
  * `algorithm`: \['auto', 'ball\_tree', 'kd\_tree', 'brute']
  * `p`: \[1, 2, 3, 4]
* Retrieved `best_params_` and `best_score_` from the trained grid

#### 7. **Visualization**

* Plotted the **1D decision boundary** using `texture_mean`
* Used color to differentiate between predicted Malignant and Benign outcomes

---

###  Model Evaluation

* Evaluated on test data using:

  * `accuracy_score`
  * `confusion_matrix`
  * `classification_report`
* Achieved accuracy close to the best cross-validated result from GridSearchCV

---

###  Decision Boundary Plot

The 1D plot shows:

* Scaled `texture_mean` values (x-axis)
* Model's predicted class (`M` or `B`)
* Green line shows **KNN decision boundary**

---

###  Requirements

* Python 3.x
* Libraries:

  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn
  ```

---

###  Notes

* Only `texture_mean` was used to keep it visually interpretable.
* For more accurate results, try using **more features** (e.g., `radius_mean`, `area_mean`) and visualizing in 2D.
* KNN is sensitive to feature scaling — standardization is crucial.

---

### 📁 Files

* `Cancer_Data.csv` — dataset
* `knn_cancer_classification.py` or notebook — code
* `README.md` — this file
