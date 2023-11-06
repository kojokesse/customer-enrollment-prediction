# customer-enrollment-prediction


```markdown
# Cart Abandonment Prediction

This project uses a machine learning model to predict whether a customer should be enrolled or not based on various
features such as age, income, price, satisfaction rating, and more.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine:
   ```shell
   git clone <repository-url>
   ```

2. Install the required Python packages:
   ```shell
   pip install pandas numpy matplotlib seaborn joblib scikit-learn gradio
   ```

3. Run the Jupyter Notebook or Python script to load the dataset, preprocess the data, train machine learning models, and create a Gradio interface for customer enrollement prediction.

4. Launch the Gradio interface:
   ```shell
   python <script_with_gradio.py>
   ```

5. Access the Gradio interface in your web browser and use it to make customer enrollement predictions.

## Project Structure

The project is structured as follows:

- `script_with_gradio.py`: Python script that contains the code for data preprocessing, model training, and Gradio interface creation.
- `customer_data.csv`: Dataset containing customer data for training and prediction.
- `decision_tree_model.joblib`: A pre-trained Decision Tree model for customer enrollement prediction.
- `README.md`: This documentation file.

## Usage

You can use the Gradio interface to make customer enrollement predictions by entering customer information such as age, income, price, satisfaction rating, location, occupation, laptop brands, frequency of use, purpose, and tech knowledge level. The interface will provide predictions based on the pre-trained model.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Note

- This project uses Gradio for building the user interface.
- The machine learning models are built using scikit-learn.
