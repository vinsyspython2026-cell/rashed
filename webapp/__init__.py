import os
from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

import pandas as pd
from webapp.etl.etl import extract, transform, predict, load
from sklearn.linear_model import LogisticRegression


# Creating an Object of the Flask App
app = Flask(__name__)

# Configure a secret key for flashing messages
app.config["SECRET_KEY"] = "your_secret_key_here"

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv", "xls", "xlsx"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
  return (
      "." in filename
      and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
  )


def run_etl_pipeline(input_file_path, output_file_path, model_path):
    print(f"\n--- Running ETL Pipeline for {input_file_path} ---")

    # 1. Extract
    new_data = extract(input_file_path)
    if new_data is None:
        print("ETL pipeline terminated due to extraction error.")
        return

    # Keep a copy of the original data to attach predictions later
    original_new_data = new_data.copy()

    # 2. Transform
    transformed_data = transform(new_data)
    if transformed_data is None:
        print("ETL pipeline terminated due to transformation error.")
        return

    # 3. Predict
    predictions, prediction_proba = predict(transformed_data, model_path)
    if predictions is None:
        print("ETL pipeline terminated due to prediction error.")
        return

    # Add predictions back to the original new_data_df for output
    original_new_data['Predicted_Loan_Status'] = predictions
    original_new_data['Prediction_Probability_Charged_Off'] = prediction_proba

    # 4. Load
    load(original_new_data, output_file_path)
    print(f"\n--- ETL Pipeline completed. Results saved to {output_file_path} ---")



# Original index route and template
@app.route("/")
def index():
  name = "Rashed"
  return render_template("index.html", getname=name)


# Route to render the dedicated upload page
@app.route("/upload", methods=["GET"])
def upload_page():
  return render_template("upload.html")


# Route to handle the actual file upload submission via POST
@app.route("/upload", methods=["POST"])
def upload_file():
  if "file" not in request.files:
    flash("No file part in the request", "error")
    return redirect(url_for("upload_page"))

  file = request.files["file"]

  if file.filename == "":
    flash("No selected file", "error")
    return redirect(url_for("upload_page"))

  if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
    flash(f"File '{filename}' successfully uploaded!", "success")
    return redirect(url_for("upload_page"))
  else:
    flash("Invalid file type. Only CSV and Excel files are allowed.", "error")
    return redirect(url_for("upload_page"))



@app.route("/run_etl", methods=["GET", "POST"])
def run_etl():
    input_csv_file = os.path.join(app.config["UPLOAD_FOLDER"], "test_data.csv")
    output_csv_file = os.path.join(app.config["UPLOAD_FOLDER"], "new_data_predictions.csv")
    model_pkl_file = os.path.join("webapp","etl", "logistic_regression_model.pkl")

    run_etl_pipeline(input_csv_file, output_csv_file, model_pkl_file)

    # Read the output CSV to display results
    output_df = pd.read_csv(output_csv_file)
    return render_template("results.html", 
                           tables=[output_df.to_html(classes='data', header="true")], 
                           titles=output_df.columns.values)




if __name__ == "__main__":
  app.run(debug=True)