# ML Engineer

## Skills

**Languages:** Python, Java, C++, C, Javascript, SQL <br>
**Libraries & Frameworks:** PyTorch, Tensorflow, Keras, Cuda, Scikit-Learn, NumPy, Pandas, OpenCV , NLTK, RLlib, TRL 
Matplotlib, Flask, Spacy, Langchain, Langgraph, Streamlit, FastAPI, Open AI Gym <br>
**Databases:** PostgreSql, MySQL, Mongodb, Neo4j, Pinecone, Milvus, Chroma <br>
**Tools and Technologies:** Version control system, Linux, Docker, Google Cloud Platform, Apache Airflow, RabbitMQ

## Education

**M.S, Artificial Intelligence** | Northeastern University (_Sep 2023_-_Dec 2025_)  
- Coursework: Pattern Recognition and Computer Vision, Foundations of AI, Algorithms, Programming Design Paradigm  
- CGPA: 3.94/4.0

**B.Tech, Computer Science and Engineering** | Amrita Vishwa Vidyapeetham (_Jul 2019_-_Jun 2023_)  
- Related Courses: Neural Networks and Deep Learning, Linear Algebra, Probability, Machine Learning  
- CGPA: 8.1/10  

## Experience

**Research Apprenticeship @ Northeastern University (_Sep 2024 - Dec 2024_)**  
- Utilized Amazon Berkeley Object Dataset to process product types and images through OpenAI CLIP VIT-B/32, generating embeddings stored in GPU-accelerated Milvus vector store.  
- Implemented BLIP Model for image caption generation and integrated with CLIP for dual-stream embedding creation.  
- Developed a bidirectional search system enabling similarity-based matching between product types and images.  

**AI Intern @ Xnode, Remote, USA (_May 2024 - Aug 2024_)**  
- Developed a Microsoft Teams bot using Microsoft Graph API and Azure Bot Framework to autonomously join meetings and extract transcripts and chats for the Xnode platform.  
- Built a scalable backend using FastAPI and RabbitMQ for asynchronous processing and implemented a distributed Map-Reduce system for efficient transcript summarization.  
- Designed a Graph RAG pipeline to process transcript data, storing insights in a Neo4j graph database and enabling intelligent querying using Azure OpenAI.  

**Computer Vision Intern @ TensorGo, Hyderabad, India (_Mar 2023 - May 2023_)**  
- Implemented real-time automated speech recognition model pipeline for Tensorgo's emYt+.  
- Segmented the speech audio into 1s chunks and utilized OpenAI's pre-trained Whisper model, optimizing speech recognition by parallel processing chunks to reduce latency by ~40%.  

## Projects

**AI-Driven Image Enhancement Using Generative AI**
- Developed an AI-powered image enhancement platform using Hugging Face's Stable Diffusion XL for background generation and ESRGAN for super-resolution.  
- Integrated YOLO model to extract positional coordinates of objects in images.  
- Applied reinforcement learning with VIZIT Score for quality assessment, achieving a 25% improvement in image quality.  

**Multimodal RAG Platform: AI-Powered Financial Document Analysis and Querying** | [Link](https://github.com/LokeshSaipureddi/Multi-Modal-RAG)
- Built a Multimodal RAG pipeline for financial document processing using OpenAI LLM with LangChain for querying.  
- Integrated Unstructured for PDF content extraction, ChromaDB for embedding storage, and developed a Streamlit frontend for real-time chat, summarization, and report generation.  
- Added RLHF with reward-model-deberta-v3-base, using user feedback to fine-tune the reward model for better responses.  

**An Agentic RAG-Based Research Assistant System**  
- Developed a system integrating web search and ArXiv to dynamically retrieve and synthesize research data, utilizing Pinecone for similarity search and AWS S3 for document storage.  
- Automated document parsing and indexing workflows using Airflow, leveraging FastAPI for backend development.  
- Built a user-friendly interface using Streamlit, enabling seamless interaction with research data and generating PDF reports.

**CNN-Powered Image Restoration with PyTorch**
- Implemented a CNN-based colorization algorithm to convert grayscale images to color, utilizing Lab color space and intensity
  channel (L) for training the model to find the ab color spaces using the intensity channel L.
- Optimized model performance with data augmentation, hyperparameter tuning, and fine-tuning pre-trained network, achieving an
  accuracy of 73%, MSE of 24, and SSIM of 0.78 indicating high-quality color reproduction.

**Generation of Human motion video from text**
- Gathered text-to-motion datasets from AMASS database, transformed the data to 3D motion representation using SMPL model
- Utilized the CLIP model to associate the image with the text and used a transformer encoder to process these features
- Developed a classifier-free diffusion model to generate videos from the features and achieved an FID score of 58%

## Research Publications  

**An Intelligent Computation Model with DMD Features for COVID Detection** | 
[Soft Computing Journal](https://www.researchgate.net/publication/375128815_An_Intelligent_Computational_Model_with_Dynamic_Mode_Decomposition_and_Attention_Features_for_COVID-19_Detection_from_CT_Scan_Images)  
- Proposed a Shallow CNN for detecting COVID-19 with CT Scan images
- Attained an accuracy of 78.6% to Transfer learning models with only 300866 trainable parameters  
- Boosted the accuracy to 92.3% with the help of Attention-driven dynamic mode decomposition processing and parametric tuning 

**Deep Feature-Based COVID Detection Using SVM** | 
[ICICC Conference](https://www.researchgate.net/publication/363883131_Deep_Feature-Based_COVID_Detection_from_CT_Scan_Images_Using_Support_Vector_Machine)
- Developed a pipeline for COVID-19 detection using machine learning models, leveraging transfer learned features from ResNet 50, Inception V3, and Efficient b7 on CT scan images
- Extracted the features from the last convolution layers of deep learning models, fed these features to ml models for classification
- Achieved an accuracy of 86.12% using a combination of InceptionV3 and SVM, with precision and recall of 83.11% and 80.44%

