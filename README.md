# ALAASKA
Adaptive Learning for All through AI-Powered Student Knowledge Assessment

**ALAASKA** is a full-stack web application that delivers intelligent, personalized tutoring experiences powered by a fine-tuned language model. Designed with a strong pedagogical foundation, ALAASKA simulates the behavior of a supportive tutor who uses micorlearning materials like flashcards, guiding questions, and mini-quizzes to promote intuitive problem solving and learner autonomy. 

## Implementation Overview

The backend, built with asynchronous Python APIs, handles secure authentication, structured session storage, and dynamic prompt injection to ensure consistent instructional behavior across chat sessions. The frontend, built in React, provides a responsive and accessible user interface optimized for both desktop and mobile devices.

<div style="text-align: center;">
  <img
  src="https://github.com/Bonam-M/ALAASKA/blob/main/frontend/src/assets/alaaska-platform.png"
  alt="ALAASKA overview"
  title="Optional title"
  width="560" 
  height="400"
  style="margin: 0 auto;">
</div>

The system supports persistent multi-session tutoring through a sidebar-based conversation manager, markdown-rendered messaging, and token-authenticated communication between the client and server. Users can log in, manage their past conversations, and engage in real-time learning conversations with the model. ALAASKA is a research project that can serves as a framework for researchers, educators, or developers building AI-powered adaptive learning tools with pedagogical constraints.  
<br>

## Installation

To run the ALAASKA application locally, clone the repository and navigate to the project root. 
- Install the required Python backend dependencies using `pip install -r requirements.txt`, and then start the FastAPI server with `uvicorn backend.main:app --reload`. 
- For the frontend, navigate to the `frontend` directory, run `npm install` to install dependencies, and start the development server using `npm run dev`. 
- Ensure you configure your `.env` files for both backend and frontend with the appropriate API keys, JWT secrets, and endpoints. 
The app runs locally at `http://localhost:3000` (frontend) and `http://localhost:8000` (backend) by default.


