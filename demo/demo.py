import os
from flask import Flask, render_template

# Dynamically resolve the absolute path to the templates directory
template_path = os.path.join(os.path.dirname(__file__), '..', 'templates')

# Initialize the Flask application with the specified template folder
app = Flask(__name__, template_folder=template_path)

@app.route('/')
def index():
    # Handle requests to the root URL.
    # 
    # Returns:
    # --------
    # render_template : flask.Response
    #     Renders the 'index.html' page containing the user interface
    return render_template('index.html')


# Entry point for running the application
if __name__ == '__main__':
    # Launch the Flask development server with debugging enabled
    app.run(debug=True)
