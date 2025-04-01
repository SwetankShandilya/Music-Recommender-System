# Music Recommender System

A hybrid music recommendation system combining collaborative filtering and content-based K-Nearest Neighbors (KNN) models, deployed as a scalable service on AWS EC2 using Docker and FastAPI.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [How to Run](#how-to-run)
5. [Models](#models)
6. [API Endpoints](#api-endpoints)
7. [Deployment](#deployment)
8. [License](#license)

## Project Overview

This project implements a hybrid music recommendation system using two types of models:
1. **Collaborative Filtering Model** - Based on user-item interactions.
2. **Content-Based Model (KNN)** - Utilizes song features from the AcousticBrainz API.

These models are combined into a hybrid system to provide more accurate and diverse music recommendations. The entire system is deployed as a containerized service on AWS EC2 using **FastAPI** and **Docker**, ensuring scalability.

## Technologies Used

- **Python**: Programming language used for implementing models and API.
- **PyTorch**: Used for the collaborative filtering model (neural network).
- **KNN (K-Nearest Neighbors)**: Used for the content-based model to match music based on features.
- **FastAPI**: Framework for serving the model as a REST API.
- **Docker**: For containerizing the application.
- **AWS EC2**: Hosting the application on a virtual server for scalable deployment.

## Installation

To run the project locally, follow these steps:

### Clone the repository

```bash
git clone https://github.com/SwetankShandilya/Music-Recommender-System.git
cd Music-Recommender-System
```

Install Dependencies
Use requirements.txt to install the required dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
Train the Models (if not pre-trained)

Use the provided Jupyter notebooks to train both the Collaborative Filtering Model and Content-Based KNN Model.

Start FastAPI server

You can start the FastAPI server using main.py.
```bash
uvicorn main:app --reload
```
Docker (Optional)

Build and run the Docker container for production-level deployment:

```bash
docker build -t music-recommender .
docker run -d -p 8000:8000 music-recommender
```
## Models
### 1. Collaborative Filtering Model
**Description**: This model uses deep neural networks built with PyTorch to learn user preferences from interaction data (e.g., play counts, likes).

**Purpose:** Recommends items based on similar user interactions.

### 2. Content-Based KNN Model
**Description:** This model utilizes K-Nearest Neighbors to recommend items based on item features such as genre, mood, etc., extracted using the AcousticBrainz API.

**Purpose:** Recommends items similar to those that the user has liked based on content characteristics.

### 3. Hybrid Model
**Description:** Combines both models to provide a well-rounded recommendation to the user. This model aims to mitigate the drawbacks of each individual model and leverage their strengths.

## API Endpoints

The FastAPI service exposes the following endpoints:

POST /recommend: Get music recommendations based on the model selection (Collaborative, Content-Based, or Hybrid).

### Request Body Example:
```bash
{
  "user_id": 123,
  "model": "hybrid"
}
```
### Response Example
```bash
{
  "recommendations": [
    {
      "track_name": "Song 1",
      "artist": "Artist A",
      "genre": "Pop"
    },
    {
      "track_name": "Song 2",
      "artist": "Artist B",
      "genre": "Rock"
    }
  ]
}
```
## Deployment
The project is deployed on AWS EC2 using Docker containers for scalable deployment. FastAPI is used to serve the model predictions through an API, allowing users to interact with the recommender system.

To deploy the system, ensure you have an EC2 instance running and Docker configured to handle requests.

### Steps for Deployment on AWS EC2

1. Set up EC2 Instance: Launch a free-tier EC2 instance on AWS.

2. Install Docker on the instance.

3. Clone the repository to the EC2 instance.

4. Build and run the Docker container for FastAPI.

```bash
docker build -t music-recommender .
docker run -d -p 8000:8000 music-recommender
```
The system will now be running on the EC2 instance, and you can interact with it using the public IP.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
