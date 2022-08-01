# Venture Funding with Deep Learning
## Summary:
You work as a risk management associate at Alphabet Soup, a venture capital firm. Alphabet Soup’s business team has asked you to create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

## Resources:

`applicant_data.csv` - a csv of applicants who have received funding from Alphabet Soup over the years. The CSV file contains a variety of information about these businesses, including whether or not they ultimately became successful

## To - Do:

Use dataset provided to create a binary classifier model that will predict whether an applicant will become a successful business.

## Instructions:
Step 1: Prepare data for use on a neural network model.
Step 2: Compile & eval a binary classification model w/ neural network.
Step 3: Optimize the neural network model.

#### Prepare the Data for Use on a Neural Network Model
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), preprocess the dataset so that you can use it to compile and evaluate the neural network model later.
1. Read the applicants_data.csv file into a Pandas DataFrame. 
2. Review the df, looking for categorical variables that will need to be encoded.
3. Drop the “EIN” (Employer Identification Number) and “NAME” columns from the DataFrame.
4. Encode the dataset’s categorical variables using OneHotEncoder, and then place the encoded variables into a new DataFrame.
5. Add the original DataFrame’s numerical variables to the df containing the encoded variables.
