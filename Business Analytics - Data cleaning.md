## Objective
The primary objective of this phase was to process the raw `bank-full.csv` dataset and transform it into a machine-learning-ready format. This required handling categorical variables, scaling numerical features, and strictly partitioning the data to prevent data leakage during model training and evaluation.
## Method
**Data Ingestion**
- The raw dataset (`bank-full.csv`) was loaded into a Pandas DataFrame. 
- The dataset utilizes a semicolon (`;`) as a delimiter, which was explicitly defined during the ingestion phase to ensure correct parsing of the 17 distinct columns.

**Target Variable Binarization**
- The original target variable, `y` (representing whether a client subscribed to a term deposit), was recorded as text. To make it compatible with mathematical models, it was mapped to a numeric format:
	- `yes` $\rightarrow 1$
	- `no` $\rightarrow 0$

**Categorical Variable Encoding**
- The dataset contained 9 nominal categorical columns: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, and `poutcome`.
- **Method Used:** One Encoding via `pd.get_dummies()`.
- **Multicollinearity Mitigation:** The parameter `drop_first=True` was applied to drop one reference category per feature. This breaks perfect multicollinearity, fulfilling the assumptions required for algorithms like Logistic Regression.
- **Type Casting:** Any resulting boolean dummy columns were explicitly cast to integer types to guarantee that the encoded features strictly of $0$ and $1$ only.

**Train-Test Partitioning**
Prior to scaling, the encoded dataset was split into training and testing subsets to ensure unbiased model evaluation
- **Split Ratio:** 80% Training / 20% Testing.
- **Reproducibility:** A fixed random seed (`random_state=42`) was utilized to guarantee consistent splits across multiple execution runs.
- **Outputs:** `train_df` and `test_df`.

**Feature Scaling (Standardization)**
The 7 continuous numeric columns (`age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`) were on vastly different scales. Left untreated, features with larger magnitudes (like `balance`) would disproportionately dominate gradient-descent or distance-based algorithms.
- **Method:** Standardization (Z-score normalization).
- **Data Leakage Prevention:** _Crucially_, the `StandardScaler` was strictly **fitted on the training data only**. Both `train_df` and `test_df` were then transformed using these derived parameters. This ensures the testing set remains a true, unseen evaluation sample, correcting previous pipeline iterations that scaled prior to splitting (which inadvertently leaked test distribution information into the training phase).

**Result**
The preprocessing pipeline successfully generated two finalized files, `train_prepared.csv` and `test_prepared.csv` exported to CSV for downstream model consumption:
## Validation and Quality Assurance
To verify the that the code is actually cleaning the data well, several checks were executed:
```
--- VERIFICATION CHECKS ---
Total missing values in Train set: 0
Total missing values in Test set: 0
Number of non-numeric columns remaining: 0

Standardization Check (Numeric Columns in Train Set):
      age  balance  day  duration  campaign  pdays  previous
mean -0.0      0.0 -0.0       0.0       0.0   -0.0      -0.0
std   1.0      1.0  1.0       1.0       1.0    1.0       1.0

Dummy Variable Check (Min/Max values of first 5 dummy columns):
     job_blue-collar  job_entrepreneur  job_housemaid  job_management  job_retired
min                0                 0              0               0            0
max                1                 1              1               1            1

Original Dataset Shape: (45211, 17)
Train Dataset Shape: (36168, 43) (80.0%)
Test Dataset Shape: (9043, 43) (20.0%)
```
1. **Missing Values:** Confirmed $0$ missing values across both training and testing sets.
2. **Data Type Consistency:** Verified the absence of `object` or `category` data types; all columns are now strictly numeric formats suitable for tensor/matrix operations.
3. **Standardization Verification:** Verified that the scaled numeric columns in the training set possess a mean $\mu \approx 0$ and standard deviation $\sigma \approx 1$.
4. **Dummy Variable Bounds:** Confirmed all binary indicator columns possess values strictly in the set $\{0, 1\}$.
5. **Dimensionality Check:** Validated that the row counts of `train_df` and `test_df` exactly correspond to the intended 80/20 split relative to the original dataset.
The resulting `.csv` files are now fully prepared for fair model training and robust validation.