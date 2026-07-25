# 🎓 Student Performance Predictor

A Machine Learning web application that predicts a student's math score based on demographic and academic information. The project is built using Python, Flask, Scikit-learn, Docker, AWS ECR, EC2, and GitHub Actions for Continuous Deployment.

---

## 🚀 Features

- Predicts student math performance using Machine Learning.
- User-friendly web interface built with Flask.
- Data preprocessing and prediction pipeline.
- Dockerized application.
- CI/CD pipeline using GitHub Actions.
- Automatic deployment to AWS EC2 using Amazon ECR.

---

## 🛠️ Tech Stack

### Machine Learning
- Python
- Scikit-learn
- Pandas
- NumPy

### Backend
- Flask

### DevOps
- Docker
- GitHub Actions
- AWS EC2
- Amazon ECR

---

## 📂 Project Structure

```
Student-Performance-Predictor
│
├── artifacts/
├── notebook/
├── src/
│   ├── components/
│   ├── pipeline/
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/
├── static/
├── .github/
│   └── workflows/
│       └── main.yaml
│
├── application.py
├── Dockerfile
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/Sumit-Avasthi/Student-Performance-Predictor.git
```

Move into the project directory

```bash
cd Student-Performance-Predictor
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Windows

```bash
venv\Scripts\activate
```

Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
python application.py
```

Open your browser

```
http://localhost:5000
```

---

## 🐳 Docker

Build Docker Image

```bash
docker build -t student-performance .
```

Run Docker Container

```bash
docker run -p 5000:5000 student-performance
```

---

## ☁️ AWS Deployment

The project uses

- Amazon EC2
- Amazon ECR
- GitHub Actions

Deployment Workflow

1. Push code to GitHub.
2. GitHub Actions builds Docker image.
3. Image is pushed to Amazon ECR.
4. EC2 pulls the latest image.
5. Docker container is restarted automatically.

---

## 📊 Input Features

- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

---

## 🎯 Output

Predicted Mathematics Score

---


## 👨‍💻 Author

**Sumit Avasthi**

GitHub: https://github.com/Sumit-Avasthi


---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub!
