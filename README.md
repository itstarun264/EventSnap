---
title: EventSnap
emoji: 📸
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# EventSnap

An AI-powered event photo sharing and real-time streaming web application.

## How to Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/new-space).
2. Choose **Docker** as the SDK (instead of Streamlit or Gradio).
3. Choose the **Blank** template.
4. Clone the Space repository or upload the files in this directory to your Hugging Face Space repository.
5. Hugging Face will automatically detect the `Dockerfile` and start building the container.

## Data Persistence Setup (Recommended)

Since Hugging Face Spaces run in containers that reset upon restart or sleeping, you should set up persistent storage:

1. **Option A: Enable Space Persistent Storage (Easiest)**
   * Go to your Space **Settings** page.
   * Under **Persistent Storage**, choose a storage tier (there is a free/dev tier available).
   * This mounts a `/data` directory automatically. EventSnap will automatically detect it and store all SQLite databases and uploaded pictures inside it, persisting them forever.

2. **Option B: Connect External PostgreSQL & Cloud Storage (Advanced)**
   * You can host your database on a free PostgreSQL host (like [Neon.tech](https://neon.tech/) or [Supabase](https://supabase.com/)).
   * Add a Space Secret (Environment Variable) in your Space settings named `DATABASE_URL` with your PostgreSQL connection URI.
   * EventSnap will connect to the PostgreSQL database automatically instead of SQLite.

## Running Locally

To run the application locally, use your virtual environment:
```bash
venv\Scripts\python.exe app.py
```
And access it at `http://127.0.0.1:5000`.
