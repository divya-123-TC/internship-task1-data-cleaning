This project is part of my internship task, where I performed *data cleaning and preprocessing* on the Titanic dataset using Python.

## Objective

To clean and prepare raw data for machine learning models by performing:

- Handling missing values
- Encoding categorical variables
- Normalizing numerical features
- Visualizing and removing outliers

## Dataset

The dataset used is the *Titanic dataset* available in the Seaborn library.

## Tools & Libraries

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- scikit-learn (StandardScaler)

1. *Loaded the dataset* and explored basic information (null values, data types).
2. *Handled missing values*:
   - Filled missing age with the mean.
   - Filled missing embarked with the mode.
   - Dropped rows with missing deck values.
3. *Encoded categorical features*:
   - Converted sex to 0 (male) and 1 (female).
   - Used one-hot encoding for embarked.
4. *Normalized* numerical features (age and fare) using StandardScaler.
5. *Visualized outliers* using boxplots.
6. *Removed outliers* from the fare column using the IQR method.

## Output

- A cleaned and preprocessed DataFrame ready for further machine learning tasks.
