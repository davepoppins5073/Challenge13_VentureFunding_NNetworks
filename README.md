# Venture Funding with Deep Learning
## User Story:
You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team has asked you to create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

---

## Code

## Instructions:
Step 1: Prepare data for use on a neural network model.
Step 2: Compile & eval a binary classification model w/ neural network.
Step 3: Optimize the neural network model.


### Resources:

`applicant_data.csv` - a csv of applicants who have received funding from Alphabet Soup over the years. The CSV file contains a variety of information about these businesses, including whether or not they ultimately became successful. This data provided will be used to create a binary classifier model that will predict whether an applicant will become a successful business.

```python
# Read the applicants_data.csv file from the Resources folder into a Pandas DataFrame
applicant_data_df = pd.read_csv(
    'applicants_data.csv',
    encoding='utf-8', 
    index_col=None)

# Review the DataFrame
display(applicant_data_df.head())
```


### Imports
```python
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
```

### Data Prep
1. Review the df, looking for categorical variables that will need to be encoded.
2. Drop the “EIN” (Employer Identification Number) and “NAME” columns from the `df`.


#### 
```python

# Drop the 'EIN' and 'NAME' columns from the DataFrame
applicant_data_df = applicant_data_df.drop(
    columns=['EIN', 'NAME']
    )
```

3. Encode the dataset’s categorical variables using OneHotEncoder, and then place the encoded variables into a new `df`.
4. Add the original DataFrame’s numerical variables to the `df` containing the encoded variables.

```python
# Create a list of categorical variables 
print(f"When we display our dataframe with dtypes, the categorical variables are objects")

categorical_variables = list(applicant_data_df.dtypes[applicant_data_df.dtypes=='object'].index)

# Create a OneHotEncoder instance
enc = OneHotEncoder(sparse=False)

# Encode the categorcal variables using OneHotEncoder
encoded_data = enc.fit_transform(applicant_data_df[categorical_variables])

```
