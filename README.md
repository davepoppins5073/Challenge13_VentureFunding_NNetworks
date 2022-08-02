# Venture Funding with Deep Learning
![13-4-challenge-image](https://user-images.githubusercontent.com/101449950/182268374-34476611-3ca5-403d-9c50-9a912c1024c4.png)


## User Story:
You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team has asked you to create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

---
# About this repo: 

<img width="733" alt="Screen Shot 2022-08-01 at 8 53 29 PM" src="https://user-images.githubusercontent.com/101449950/182268987-6c700b0c-0bc5-4ace-9602-f8467c678fd1.png">

<b>Resource Folder:</b> This has the data-set that is central to this effort `applicant_data.csv`

<img width="588" alt="Screen Shot 2022-08-01 at 8 52 47 PM" src="https://user-images.githubusercontent.com/101449950/182269143-a53dbac9-6012-4106-9e91-0f86857b9901.png">

<b>Contents Folder:</b> this folder has all the HDF5 file generated:


---
# Code

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

## Part 1: Data Prep
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
## Part 2: Compile & Evaluate a Binary Classification Model Using a Neural Network
### Step 1:

Create a deep neural network by assigning
1. the number of input features,
2. the number of layers
3. the number of neurons on each layer using Tensorflow’s Keras.

```python
# Define the number of neurons in the output layer
number_output_neurons = 1

# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1 = (number_input_features + 1) // 2

# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2 =  (hidden_nodes_layer1) // 2

# Create the Sequential model instance
nn = Sequential()

# Add the first hidden layer
nn.add(
    Dense(
        units=hidden_nodes_layer1,
        activation='relu',
        input_dim=number_input_features
    )
)
# Add the second hidden layer
nn.add(
    Dense(
        units=hidden_nodes_layer2,
        activation='relu'
    )
)
# Add the output layer to the model specifying the number of output neurons and activation function
nn.add(
    Dense(
        units=1,
        activation='sigmoid'
    )
)

```


#### Compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric.
```python
# Compile the Sequential model
nn.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# Fit the model using 50 epochs and the training data
fit_model = nn.fit(X_train_scaled, y_train, epochs=50)
```
#### Evaluate the model using the test data to determine the model’s loss and accuracy.
```python
# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
268/268 - 0s - loss: 0.5541 - accuracy: 0.7299 - 475ms/epoch - 2ms/step
Loss: 0.5541415810585022, Accuracy: 0.729912519454956
```
#### Save and export your model to an HDF5 file, and name the file AlphabetSoup.h5.
```python
# Set the model's file path
file_path = "/content/DP_AlphabetSoup.h5"

# Export your model to a HDF5 file
nn.save_weights(file_path)
from google.colab import drive
```

### Part 3: Optimize the Neural Network Model
<b>Comparing Accuracy Scores</b>
1. Alternative Model 1 - Accuracy: 0.53
2. Alternative Model 2 - Accuracy: 0.73

**Note**: Perfect accuracy has a value of 1. Optimizing your model for a predictive needs an accuracy to be as close to 1 as possible. 

For my first Alternative model I reducted the number of features being looked at through PCA (Principle component analysis). More specifically we reduced the components and used standard scaler to normalize the data and continued with the same process used in the original model. The PCA method didnt improve anything but forced me to give thouight to how many components were necessary. 

The Second Alternative Model I had an extra layer this didnt improve the accuracy much as the accuracy score for the original model was Accuracy: 0.729912519454956. I used the same process as the previous code  so there wasnt as much need to  go over the steps here again
