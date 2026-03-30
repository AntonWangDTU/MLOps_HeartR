# MLOps Heart Disease Predictor

A simple **FastAPI** application that predicts the risk of heart disease based on patient data. Includes a Docker setup so you can run it easily anywhere.  

---

## Features

- Predict heart disease risk using 7 key features:
  - `age`, `sex`, `cp`, `trestbps`, `chol`, `thalach`, `exang`
- Simple web interface to input values
- Returns prediction and probability
- Runs in a Docker container for easy deployment

---

## Requirements

- [Docker](https://www.docker.com/get-started) installed
- Optional: [Python 3.12](https://www.python.org/downloads/) if you want to run locally without Docker

---

## Build and Run with Docker

1. **Clone the repository**

```bash
git clone git@github-anton:AntonWangDTU/MLOps_HeartR.git
cd MLOps_HeartR
```


2. **Build the Docker image** 

```bash
docker build -t heart-predictor .
``` 

3. **Run the container** 

```bash
docker run -p 8000:8000 heart-predictor
```

4. **Access the app** 

Open your browser at [http://localhost:8000](http://localhost:8000)  to use the web interface.



