# Identifying Misinformation in News Articles Using NLP and Machine Learning
## Project Overview

Our project is titled "Identifying Misinformation in News Articles Using NLP and Machine Learning." We aim to use natural language processing techniques and machine learning models to build a system that can detect and classify fake news articles. We'll preprocess news data and apply machine learning models to classify articles as real or fake, with plans to deploy the model as a web application for user interaction


## Introduction

In today’s rapidly evolving digital landscape, the proliferation of fake news poses a significant threat to public trust, democratic processes, and social cohesion. Fake news, defined as deliberately misleading or false information presented as news, has the potential to cause widespread harm by manipulating public opinion and fostering misinformation. Despite efforts from human fact-checkers and media platforms, the sheer volume of information makes it impossible to manually verify every article. Additionally, current automated solutions for detecting misinformation often struggle with domain-specific nuances, context awareness, and scalability.
This project aims to develop a robust, scalable system using Natural Language Processing (NLP) and Machine Learning (ML) to detect misinformation in news articles. The proposed system will analyze textual patterns and linguistic features across a large dataset of both reliable and unreliable news sources. By leveraging NLP techniques like contextual embeddings (BERT) and machine learning models, the system will classify articles into "reliable" or "misinformation" categories. The model will address key challenges such as data imbalance, domain generalization, and the need for real-time performance in large-scale news environments.
The focus is on enhancing current methods by incorporating context-awareness and improving the accuracy of identifying subtle and complex misinformation, especially in politically charged or sensitive topics.

## Literature Review

Research on fake news detection has gained significant attention in recent years. Different
approaches combine content analysis (based on text) with contextual analysis (based on how
news spreads in social networks) to improve detection accuracy.

### Article 1: "Fake News Detection on Social Media: A Data Mining Perspective"

#### Objective: 
This paper explores how combining content analysis (NLP) with context
analysis (social propagation patterns) can improve the detection of fake news.
#### Key Findings: 
The study suggests that using Bag-of-Words and TF-IDF alongside social
context features such as how news is shared can significantly improve fake news detection.
### Article 2: "Fake News Detection on Social Media Using Geometric Deep Learning"
#### Objective: 
This paper proposes the use of Graph Convolutional Networks (GCNs) to
detect fake news by modeling the way news spreads across social networks.
#### Key Findings: 
The study found that fake news spreads differently than real news, and
using GCNs achieved an accuracy of 92.7% in detecting fake news by analyzing these
patterns.

## Datasets
The dataset is a collection of news articles available through the link [Dataset](https://huggingface.co/adyamp).

  Number of files: 2 
  Total size: 116.37 MB

**Document 1: Fake.csv**

  Size: 62.79MB
  
  Number of rows: 23,481
  
  Number of columns: 4

**Document 2: True.csv**

  Size: 53.58MB
  
  Number of rows: 21,417
  
  Number of columns: 4

   
# Data 

``` python
fake_data.head()
```

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/1.png)

``` python
fake_data.info()
```

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/2.png)
``` python
read_data.head()
```
![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/3.png)
``` python
read_data.info()
```
![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/4.png)

# Data Preprocessing

## 1. Concatenation of Datasets:
The dataset is initially segregated into two CSV files: **Fake.csv** and **Real.csv**. The process involves merging them and introducing a target variable: assigning 0 to Fake News and 1 to True/Real News.


## 2.Checking for Null Values:
Upon combining the dataset, it is confirmed that there are no null values present.


## Text Cleaning Process

 ## 1. Noise Removal:
   Stripped text of formatting, including HTML tags, special characters, and emojis, to focus solely on the textual content.

 ## 2. Lowercasing:
   Converted all text to lowercase to maintain consistency; for example, "Hello" and "hello" were treated as the same word for analysis.

 ## 3. Punctuation Removal:
   Discarded punctuation marks from the text, as they typically don't convey significant semantic meaning.

 ## 4. Stopwords Removal:
   Removed common words like "is", "and", "the", etc., known as stopwords, which often don't contribute much to the text's meaning.

 ## 5.Numbers Removal:
   Eliminated numerical values or converted them into textual representations, as they might not be relevant for the analysis.

# Preprocessing Steps in NLP

## 1. Tokenization:

**Sentence Tokenization:** Divided the text into individual sentences.

**Word Tokenization:** Split the text into individual words or tokens.

## 2. Stemming:

Reduced words to their base form (e.g., "running" to "run").

## 3. Lemmatization:

Mapped contextually linked words with similar meanings to a single word (e.g., "better" to "good").

## 4. Word2Vec Embeddings:

Word2Vec maps words into a dense vector space, capturing semantic meanings based on their context in the corpus.
Converted sentences to their average word vector representations using a Word2Vec model.


# Exploratory Data Analysis

###  1. Frequency of Subjects 

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/dv1.png)

T﻿his bar graph illustrates the distribution of subject frequencies within the dataset, highlighting "US News" and "Politics" as the most prevalent subjects. 

### 2. Donut Chart of Categories

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/dv2.png)
 
The Donut chart illustrates the distribution of real and fake data in the dataset, showcasing a fair balance between the two categories.

### 3. Timeline Visualization of  Fake and Real News Articles:


![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/dv3.png)
 

The above graph demonstrates the connection between article counts and their release dates. Notably, there is a significant increase in articles during May and June of 2016. Additionally, a spike in articles is observed towards late 2017.

### 4. Distribution of Article Lengths by Category

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/5.png)
 
 


### Word Cloud 

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/6.png)

From the word cloud, we can observe that the most commonly used words in fake news articles seem to be "said", "Donald Trump", "American", "people", "that", "according", "support", "action", "women”,  "Hillary Clinton".

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/wc2.png)

From the  above visualization , it's evident that articles categorized as true tend to have a greater average word length compared to those categorized as fake. Typically, individuals fabricating information tend to employ numerous articles and prepositions. Conversely, individuals conveying truthful information often exhibit greater articulateness and conciseness in their language use . Therefore, the conclusion drawn from the visualization, indicating that true news articles have a shorter average word length, appears to be valid.

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/wc3.png)

The above graphs reveal that fake texts generally contain more characters and words per article, thereby supporting the hypothesis established by the preceding visualization.

### 6.  N GRAM Analysis

N-gram analysis involves breaking down text into sequences of N consecutive words and then analyzing the frequency and patterns of these sequences. This technique is widely used in natural language processing tasks such as language modeling, text generation, and sentiment analysis. By examining the occurrence and co-occurrence of these sequences, N-gram analysis can provide insights into the structure and patterns of language, helping to identify common phrases, expressions within a body of text.

#### Bi-Gram Analysis

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/BiGram.png)


Upon analyzing **bi-gram** in news titles, distinct patterns emerge between fake and true news datasets.

In fake news titles, 'Donald Trump' appears 547 times, showing a strong focus on sensationalism or possible political bias. This frequent mention suggests an aim to attract attention or stir controversy in fabricated stories. 'White House' follows closely with 268 appearances, reinforcing the theme of political intrigue or manipulation.

Conversely, in true news titles, 'White House' dominates with 734 appearances, highlighting the importance of political coverage and government affairs in real news. 'North Korea' comes second with 578 appearances, indicating a significant focus on international relations and geopolitical developments.


#### Tri-Gram Analysis

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/TRiGram.png)

Upon analyzing **tri-gram** in news titles, distinct patterns emerge between fake and true news datasets.

Occurrences of phrases such as 'Boiler Room EP' and 'Black Lives Matter' in fake news tri-grams suggest a tendency towards sensationalism or subjective interpretation of events. These phrases, often used to evoke strong emotions or attract attention, indicate a preference within fabricated stories for dramatic or controversial elements over factual accuracy.

On the other hand, tri-grams found in genuine news articles containing phrases like 'White House says' and 'Trump says he' demonstrate a more objective approach to reporting, focusing on statements from credible sources. By attributing information to authoritative figures or institutions, fake  news sources aim to provide readers with reliable and verifiable information, maintaining journalistic integrity.

Tri-grams in fake news articles may include misleading phrases like 'To Vote For', aiming to manipulate readers' perceptions or influence their behavior. This underscores the deceptive nature inherent in fabricated news narratives.

# 9. Model Development 


## Parallel Processing :

  Parallel processing is used to increase the computational speed of computer systems by performing multiple data-processing operations simultaneously. 
  To expedite text preprocessing, the data is divided into smaller chunks.
  Each chunk undergoes preprocessing tasks independently and simultaneously using multiple CPU cores.
  This parallel processing significantly speeds up the overall preprocessing workflow, especially for large datasets.

## Word2Vec Embeddings: 

Word2Vec is a widely used method in natural language processing (NLP) that allows words to be represented as vectors in a continuous vector space. Word2Vec is an effort to map words to high-dimensional vectors to capture the semantic relationships between words . Words with similar meanings should have similar vector representations, according to the main principle of Word2Vec .

## Average Word Vector Representation:

Average word vector representation is a method used in natural language processing to convert sentences or text sequences into fixed-length numerical vectors. This technique involves first representing each word in the sentence as a word embedding vector, where similar words are closer in vector space. These word embeddings are then averaged element-wise to create a single vector representation for the entire sentence, capturing its semantic meaning based on the meanings of its constituent words. This approach allows for the encoding of variable-length text inputs into a consistent format suitable for machine learning algorithms, facilitating tasks such as text classification, sentiment analysis, and document clustering.

# Model Preparation

Since this task involves classification, the chosen models are classifiers, including **Logistic Regression**, **Linear SVM**, **Random Forest**, **Decision Tree**, and **Gradient Boosting**. Google Colab served as the development environment due to its convenience in importing packages. For testing and validation, 20% and 10% of the dataset were utilized

## 10. Classifier Evaluation:

 Created a function  which evaluates the performance of multiple classifiers using word embeddings (Word2Vec) on a test dataset. It trains each classifier, makes predictions on the test dataset and calculates evaluation metrics and stores them in a dictionary. 

##  Assessment metrics:

1. F1 Score (harmonic mean of precision and recall)
2. Confusion matrix (visualizes accurate and incorrect predictions)
3. ROC curve and Area Under the Curve (AUC) to measure version overall performance.
4. Stores the results for each version with Word2Vec capabilities in a dictionary named **res**.

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/M1.png)

 From the above output Logistic Regression, Linear SVM, and Random Forest classifiers exhibit strong overall performance, as evidenced by their high F1 scores exceeding 0.96. These classifiers demonstrate effective classification ability, with minimal misclassifications as indicated by their respective confusion matrices. While Decision Tree and Gradient Boosting classifiers show slightly lower F1 scores, around 0.92 and 0.95 respectively, they still demonstrate respectable performance, with slightly higher false positive rates. Overall, the results indicate that Logistic Regression, Linear SVM, and Random Forest classifiers perform exceptionally well in this classification task using Word2Vec embeddings, with Decision Tree and Gradient Boosting classifiers offering competitive performance.

## ROC Curve Visualization:

![image](https://github.com/Shivateja3/Capstone-Project-606/blob/main/Project_Images/F1.png)

The ROC curve is a graphical representation of the True Positive Rate (TPR) against the False Positive Rate (FPR) , we can see the the graph of Logistic regression , Linear svm and Random forest approaching the top-left corner of the plot, suggesting strong discriminative ability between positive and negative instances. 

The AUC represents the area under the ROC curve and summarizes the performance of the classifier across all possible threshold settings.

AUC ranges from 0 to 1, where a higher value indicates better performance. 

The AUC values for Logistic Regression, Linear SVM, and Random Forest are around 0.96, indicating their high True Positive Rate (TPR) and relatively low False Positive Rate (FPR).

 # 11. Deploying the Model Using Streamlit

 Streamlit was employed to develop a web application, This includes a text input field and a submit button, enabling users to input text for analysis. Upon submission, the model processes it using pre-trained models and provides real-time predictions on whether the news is fake or real. This interactive functionality not only enhances user accessibility to the model but also offers immediate feedback on the authenticity of news content, demonstrating the practical utility of machine learning models in real-world scenarios.

## Fake News 

![image](https://github.com/UMBC-1/Capstone-Project/assets/119750555/8d4cd959-18ac-4df2-8e1d-2fd158408bef)


## Real News

![true](https://github.com/UMBC-1/Capstone-Project/assets/119750555/b47b2ea2-20f2-41b5-832b-13349c15efa3)


# 12.Conclusion

## Limitations 

1. Relying solely on Word2Vec embeddings may limit the system's adaptability to evolving forms of fake news that differ significantly from the training data. This could lead to reduced performance in detecting novel deceptive tactics.

2. The computational resources required for training and maintaining sophisticated models, especially at scale for real-world deployment, can be substantial and costly. This includes the need for robust infrastructure and significant financial investments to ensure efficient operation and scalability.

3. Ensuring data quality and addressing biases in training data are essential for model fairness and generalizability. Failure to adequately address these concerns may result in biased predictions and undermine the model's effectiveness in diverse real-world scenarios.

## Future Work

1. Experiment with ensemble methods to combine multiple models (e.g., logistic regression, random forest) for enhanced predictive performance and model robustness against various types of deceptive content.

2. Develop mechanisms for real-time monitoring and continuous model updates with new data to adapt to evolving fake news patterns and ensure model relevancy over time. This iterative approach contributes to the ongoing battle against misinformation in the digital landscape.

