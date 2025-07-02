# Agentic-Arbitration-Prediction
Estimating Arbitration Amount & Dispute Classification
This project, developed under the guidance of Prof. Anurag Agarwal (IIM Ahmedabad), focuses on building an end-to-end intelligent data extraction and analysis system for publicly available legal documents and articles related to infrastructure arbitration cases in India. It aims to classify arbitration disputes and estimate potential settlement amounts by integrating web scraping, natural language processing (NLP), and machine learning (ML) to derive actionable insights from unstructured legal data.

🌟 Project Overview
The legal domain, particularly arbitration, involves vast amounts of unstructured data. Manually sifting through legal documents for case classification and settlement estimation is time-consuming and prone to human error. This project addresses these challenges by automating the extraction of critical information and applying advanced AI models to predict dispute outcomes and settlement figures.

🚀 Approach
Our methodology involved a multi-stage, AI-driven pipeline:

Data Collection: Automated web crawling and scraping of legal dispute information from relevant online sources.

Data Annotation: Utilizing Label-Studio for classifying and labeling key features in the collected data to prepare for model training.

AI Modeling: Implementing an Agentic Retrieval Augmented Generation (RAG) model, enhanced with the DeepSeek API, for context-aware legal entity recognition, case classification, and settlement amount prediction.

🧩 Key Components & Features
The project is structured around several interconnected modules:

1. Case-Based Legal Data Scraping
Scraped and processed over 50+ infrastructure arbitration cases using case identifiers across India Kanoon and tribunal websites.

Utilized Playwright and Google Search API to automate access and retrieval of relevant case documents, judgments, and metadata.

Achieved a document collection efficiency improvement of ~65% compared to manual searches.

2. Named Entity Recognition (NER) for Legal Metadata Extraction
Applied a Retrieval-Augmented Generation (RAG) pipeline, powered by DeepSeek API, for context-aware legal entity recognition.

Extracted over 2,000+ legal entities including parties, jurisdictions, project types, and dispute categories from unstructured text.

Fine-tuned a domain-specific BERT model for legal NER with an F1 score of 0.71 and classification accuracy of 50.6%.

3. Sector-wise Keyword Clustering
Clustered 1,200+ extracted keywords using semantic embeddings (SBERT) and k-means clustering.

Segregated cases into 6+ infrastructure sectors (e.g., transportation, water, energy, telecom, construction, and ports).

Improved sector tagging accuracy by ~30% using this clustering step before further enrichment.

4. Data Source Expansion via Focused and Generalized Web Crawling
Designed a hybrid crawl strategy combining cluster-aware focused scraping with broad web crawling to capture additional legal commentaries.

Gathered over 1,500+ supplementary legal documents and articles, adding contextual diversity to case histories.

5. Re-application of RAG/NER on Expanded Dataset
Re-ran the DeepSeek-based pipeline on the enriched dataset to extract deeper insights, temporal data, and inter-case references.

Generated over 1,000+ structured JSON schemas, encoding timelines, participants, legal issues, and resolutions.

6. Multi-Agent AI Workflow for Preprocessing & Annotation
Developed a set of agentic Python scripts to automate document cleaning, entity resolution, and annotation file generation.

Produced over 800 Label Studio-ready annotated samples, enabling supervised fine-tuning of legal AI models.

7. Predictive Modeling of Arbitral Awards
Used tribunal-level features (e.g., arbitrator experience, sector, jurisdiction) to train linear and lasso regression models.

Achieved Mean Absolute Error (MAE) of 12.65 in predicting arbitral award amounts.

Generated feature importance maps, highlighting key drivers of award size (e.g., contract value, delay duration).

📊 Results & Achievements
The project successfully developed a comprehensive system for legal data analysis. Key outcomes include:

Efficient Data Collection: Automated retrieval of 50+ arbitration cases, improving document collection efficiency by ~65%.

Robust Entity Extraction: Built a DeepSeek-powered RAG pipeline for legal NER, extracting 2,000+ entities with an F1 score of 0.71 using a fine-tuned legal BERT model.

Enhanced Classification: Clustered 1,200+ keywords using SBERT & K-means to classify cases into 6+ infrastructure sectors, enhancing sector-tagging accuracy by ~30%.

Contextual Enrichment: Expanded the dataset with 1,500+ legal articles via hybrid crawling, enabling deeper contextual analysis and generation of 1,000+ structured JSON schemas.

Automated Annotation: Designed multi-agent workflows for automated annotation and cleaning, generating 800+ Label Studio-ready samples.

Predictive Insights: Modeled arbitral award prediction using tribunal-level features, achieving an MAE of 12.65 and identifying key award determinants.

🛠️ Technologies Used
Languages: Python, R

Web Scraping: Playwright, Google Search API

AI/NLP: DeepSeek API, BERT, SBERT, Retrieval Augmented Generation (RAG)

Data Annotation: Label-Studio

Machine Learning: K-means Clustering, Linear Regression, Lasso Regression
