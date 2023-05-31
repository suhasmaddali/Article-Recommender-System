# 📚 Article Recommender System

[![](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=darkgreen)](https://www.python.org)  [![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org) [![](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/stable/) [![](https://img.shields.io/badge/SciPy-654FF0?style=for-the-badge&logo=SciPy&logoColor=white)](https://www.scipy.org) [![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org) [![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)  [![](https://img.shields.io/badge/Plotly-239120?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com) [![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io) [![](https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com)

## Introduction

We see that articles are present almost everywhere which convey a lot of information such as the news, events and information about latest technologies and innovation. Reader engagement with the articles has increased to a large extent compared to the 20th century where currently, they have access to the internet and people are able to gather content and read well.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/Article%20Recommender%20System%20Image.jpg" width = "750" />

## Challenges

When a reader is going through an article, they might sometimes get bored and feel like not reading because the present article contains information that is beyond the comprehension of a reader to understand and interpret text compared to previous articles. Hence their interest would drop as a result of articles being dissimilar to the already read articles. 

One of the key things to note is that a user or a reader would be more engaged in those articles that are similar to the ones read earlier. Therefore, if we could come up with a technology that recommends users' articles, this would ensure that the reader is engaged and is likely going to read other articles as well. 

## Exploratory Data Analysis

This step plays a crucial role in machine learning by uncovering diverse patterns and trends within the data. By doing so, we gain valuable insights into the data's quality, including the presence of missing information, outliers, or inaccurate labels. Furthermore, this information enables us to effectively communicate specific data-related actions to the business, empowering them to take appropriate measures accordingly.

To visualize the null values in the dataset, we will employ the missingno plot. While some features, like text and others, do not have any missing values, there are certain sets of features that do contain missing values. Considering that these particular features do not contribute to the predictive performance of machine learning models, it is advisable to remove them from the analysis.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/Missingno%20plot.jpg"/>

As part of our analysis, we will visualize the time stamp information of the articles. Upon examination, we have observed that there are numerous instances where the time stamps are similar. Consequently, a significant portion of our data comprises articles with similar time stamps, allowing us to focus on this subset of data for further investigation.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/Timestamp%20Histplots.jpg"/>

The majority of the articles are presented as links, while a smaller proportion consists of videos and other formats. However, the predominant format for the articles is HTML, providing us with a consistent structure for analysis and processing.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/Content%20type%20histplot.jpg"/>

The majority of the articles are written in English, with a smaller number available in other languages. As a result, it is advantageous for us to prioritize our attention and efforts towards analyzing the English language articles to maximize the potential benefits and insights obtained.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/Language%20barplot.jpg"/>

Wordclouds provide a visual depiction of word frequencies within our text corpus, offering valuable insights into the relative occurrence of specific words. The size of each word in the wordcloud reflects its frequency in the text. By utilizing wordcloud plots, we can effectively analyze word occurrences. Notably, our titles frequently include terms such as "Google," "Machine Learning," and other related technologies, indicating their prominent presence.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/Title%20wordcloud.jpg"/>

We will analyze the entire text to gain insights into its sentiment, while also paying attention to the most recent words that appear. It is often observed that certain words like "one" and "will" occur frequently in comparison to others.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/Text%20wordcloud.jpg"/>

To cluster and recommend articles, we will utilize principal component analysis (PCA) to identify the key components. This approach allows us to determine the percentage of variance explained by the models. Upon analysis, we observe that a significant number of elements contribute to approximately 90 percent of the total variance. Consequently, we conclude that reducing the number of components extensively is unnecessary.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/PCA%20variance%20explained.jpg"/>

To obtain a high percentage of variance explained, a PCA analysis on a tf-idf text array typically requires at least 1000 or more components. This suggests that a considerable number of components is necessary to accurately represent the data points and minimize reconstruction errors.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/PCA%20tfidf.jpg"/>

The elbow method is used to identify the optimal number of clusters for a k-means clustering model. In our case, we have determined that the most suitable value for k is 11. This decision is based on observing that beyond this number, the reduction in total inertia becomes marginal compared to the significant decrease observed when transitioning from 1 cluster to 11 clusters.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/k%20means%20clustering.jpg"/>

The 2-dimensional visualization of the data using PCA and the k-means clustering model reveals that a considerable number of clusters are accurately classified. However, there is room for improvement in refining certain clusters for better accuracy.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/k%20means%202d%20plot.jpg"/>

We will now examine the 3D representation of the clusters formed using the k-means clustering approach. It is evident that some points lie significantly further away from the central data points. Nevertheless, the clustering approach demonstrates satisfactory performance in classifying and grouping the data points into distinct clusters.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/3d%20plot%20k%20means.jpg"/>

We will analyze the TF-IDF vectorization and apply PCA to visualize the clusters in a 2-dimensional view. The classification and division of data points into multiple clusters are generally successful. Notably, the PCA results after employing TF-IDF vectorization differ from those obtained using the previous vectorizer we utilized.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/tsne%202d%20plots.jpg"/>

Next, we will examine the 3D representation of the data points. It is notable that the data points appear to be closely clustered together after performing PCA with dimensionality reduction. The number of outliers is significantly reduced compared to the previous method. Overall, the k-means clustering approach with the implementation of TF-IDF vectorization demonstrates strong performance in accurately classifying the data points.

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/tfidf%203d%20plot.jpg"/>

In a similar vein, we would now be using the kernel PCA to determine the best ways to cluster with k-means clustering model. 

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/Kernel%20PCA%202d%20plots.jpg"/>

<img src = "https://github.com/suhasmaddali/Article-Recommender-System/blob/main/Images/Kernel%20PCA%203d%20plots.jpg"/>

## 👉 Directions to download the repository and run the notebook 

This is for the Washington Bike Demand Prediction repository. But the same steps could be followed for this repository. 

1. You'll have to download and install Git which could be used for cloning the repositories that are present. The link to download Git is https://git-scm.com/downloads.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(14).png" width = "600"/>
 
2. Once "Git" is downloaded and installed, you'll have to right-click on the location where you would like to download this repository. I would like to store it in the "Git Folder" location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(15).png" width = "600" />

3. If you have successfully installed Git, you'll get an option called "Gitbash Here" when you right-click on a particular location. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(16).png" width = "600" />


4. Once the Gitbash terminal opens, you'll need to write "Git clone" and then paste the link to the repository.
 
&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(18).png" width = "600" />

5. The link to the repository can be found when you click on "Code" (Green button) and then, there would be an HTML link just below. Therefore, the command to download a particular repository should be "Git clone HTML" where the HTML is replaced by the link to this repository. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(17).png" width = "600" />

6. After successfully downloading the repository, there should be a folder with the name of the repository as can be seen below.

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(19).png" width = "600" />

7. Once the repository is downloaded, go to the start button and search for "Anaconda Prompt" if you have anaconda installed. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(20).png" width = "600" />

8. Later, open the Jupyter notebook by writing "Jupyter notebook" in the Anaconda prompt. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(21).png" width = "600" />

9. Now the following would open with a list of directories. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(22).png" width = "600" />

10. Search for the location where you have downloaded the repository. Be sure to open that folder. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(12).png" width = "600" />

11. You might now run the .ipynb files present in the repository to open the notebook and the python code present in it. 

&emsp;&emsp; <img src = "https://github.com/suhasmaddali/Images/blob/main/Screenshot%20(13).png" width = "600" />

That's it, you should be able to read the code now. Thanks. 
