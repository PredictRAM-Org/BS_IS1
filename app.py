import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to perform linear regression and predict income statement item
def predict_income_item(data, col1, col2, percentage_change):
    # Extract features (X) and target variable (y)
    X = data[[col1]]
    y = data[col2]

    # Initialize and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict income statement item based on percentage change in input2
    new_input2 = data[col2].mean() * (1 + percentage_change / 100)
    predicted_item = model.predict([[new_input2]])

    return predicted_item[0]

# Main function for the Streamlit app
def main():
    # Set page title
    st.title("Income Statement Item Prediction App")

    # Upload Excel file
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

    if uploaded_file is not None:
        # Read the Excel file into a DataFrame
        data = pd.read_excel(uploaded_file)

        # Display the DataFrame
        st.dataframe(data)

        # Select columns for linear regression
        col1 = st.selectbox("Select Column 1", data.columns)
        col2 = st.selectbox("Select Column 2 (Target Variable)", data.columns)

        # Input percentage change
        percentage_change = st.number_input("Enter Percentage Change in Input2", min_value=-100.0, max_value=100.0, value=0.0)

        # Perform linear regression and make predictions
        predicted_value = predict_income_item(data, col1, col2, percentage_change)

        # Display the predicted value
        st.subheader("Predicted Income Statement Item:")
        st.write(predicted_value)

        # Plot the predicted item point on the graph
        plt.scatter(data[col1], data[col2], label="Original Data")
        plt.scatter(new_input2, predicted_value, color='red', label="Predicted Point")
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.title("Linear Regression Prediction")
        plt.legend()
        st.pyplot()

# Run the app
if __name__ == "__main__":
    main()
