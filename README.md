# MaizeIQ Kenya — Maize Price Prediction System

## Project Structure
```
maize_system/
├── app.py                        # Flask backend (main entry point)
├── best_model.pkl                # Trained Gradient Boosting model
├── model_features.pkl            # Feature list for the model
├── merged_maize_weather.csv      # Dataset (maize + weather merged)
├── maize.db                      # SQLite database (auto-created on first run)
├── requirements.txt
├── static/
│   ├── css/main.css
│   └── js/main.js
└── templates/
    ├── base.html                 # Base layout with fixed navbar
    ├── home.html                 # Landing page
    ├── prediction.html           # Prediction form
    ├── historical.html           # Historical data + chart + download
    ├── about.html                # About the project
    ├── login.html
    ├── register.html
    ├── user_dashboard.html
    └── admin_dashboard.html
```

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://127.0.0.1:5000
```

## Default Admin Account
- Email: admin@maize.co.ke
- Password: admin123

## Pages
| URL            | Page                          |
|----------------|-------------------------------|
| /home          | Landing page                  |
| /prediction    | Price prediction form         |
| /historical    | Historical data + CSV download|
| /about         | About the project             |
| /login         | Login                         |
| /register      | Register                      |
| /dashboard     | User or Admin dashboard       |

## API Endpoints
| Endpoint                  | Method | Description                        |
|---------------------------|--------|------------------------------------|
| /api/predict              | POST   | Returns predicted price            |
| /api/historical           | GET    | Returns filtered historical data   |
| /api/historical/download  | GET    | Downloads filtered CSV             |
| /api/chart-data           | GET    | Monthly aggregated price for chart |
| /admin/delete-user/<id>   | POST   | Admin: delete a user               |

## Deployment (Ubuntu Server)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```
