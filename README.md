# Inkmaster

Inkmaster is designed to help calligraphy beginners learn and improve more effectively by leveraging multiple AI techniques to provide personalized writing guidance.

## 📌 Overview

This project builds an end-to-end pipeline for calligraphy learning, integrating computer vision, image retrieval, and large language models:

- Use a Vision Transformer (ViT) to recognize calligraphy master styles  
- Apply skeletonization to extract character structure  
- Retrieve similar characters via image-based RAG  
- Generate personalized feedback using an LLM  

## 🚀 Pipeline

1. Input a handwritten calligraphy image  
2. Identify the master style (ViT)  
3. Extract character skeleton (Skeleton algorithm)  
4. Retrieve similar examples (Image RAG)  
5. Generate improvement suggestions (LLM)  

## 🧠 Tech Stack

- Vision Transformer (ViT)  
- Skeletonization algorithms  
- Retrieval-Augmented Generation (RAG)  
- Large Language Models (LLM)  
- Streamlit (for frontend visualization)  

## 📂 Project Structure

- `combine.ipynb`  
  Main notebook that integrates the full pipeline  

- `vit_finetune.ipynb`  
  Fine-tunes the ViT model (must be run first)  

- `embedding.ipynb`  
  Preprocessing step for image RAG (generate embeddings for selected masters)  

- `Pesudo label.ipynb`  
  Generates CSV labels mapping calligraphy images to corresponding Chinese characters  

- `combine_web_app.py`  
  Streamlit-based web application  

## ⚙️ Usage

Run the following notebook to directly see the result:

`vit_finetune.ipynb`

If you want to launch the app,

run the following code in the bash:

`python -m streamlit run combine_web_app.py`
