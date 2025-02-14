# Flask DOS Service

This project is a Flask application that simulates a Denial of Service (DOS) attack and integrates anomaly detection using the Hybrid Isolation Forest model. The application provides an endpoint to trigger the simulation of a DOS attack and alerts when an attack is detected.

## Project Structure

```
flask-dos-service
├── app
│   ├── __init__.py
│   └── routes.py
├── HIF_predict.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd flask-dos-service
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the Flask application:
   ```
   python -m flask run
   ```

2. Simulate a DOS attack by accessing the following endpoint:
   ```
   http://127.0.0.1:5000/simulate_dos
   ```

3. The application will display alerts in the console when a DOS attack is detected.

## Dependencies

- Flask
- Any other necessary libraries (as specified in `requirements.txt`)

## License

This project is licensed under the MIT License.