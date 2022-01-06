# notMNISTClassification
Course Project at UMN (CSCI5525 - Advanced Machine Learning)
NotMNIST Image classification using Variational Autoencoders and Topological Data Analysis <br>
Final report (CSCI5525_Project_Report) has a detailed explaination of the process.

## Workflow 1 (Deep Learning - Variational Autoencoder + Logistic Regression):
* Main idea is to get the compressed latent-space representation of an image from the Variational Autoencoder(VAE) and pass these features into a linear model(Logistic Regression) for the classification task.

* Convolution Block <br>
  <img src="https://user-images.githubusercontent.com/43849409/148102927-ef60c71d-e9f3-4916-b3fd-afc3e52fb531.png" width="250" height="150"/>

* VAE Architecture <br>
  <img src="https://user-images.githubusercontent.com/43849409/148102942-bd0aa195-0ca2-48c3-b684-ba824d743385.png" width="500" height="200"/>

* Training VAE
  * The training set consists of 14k grayscale images and the validation-set consists of 4.5k grayscale images.
  * Initial learning rate(LR) of 1e-4 was used along with a LRscheduler - ReduceLROnPlateau to fetch the optimal LR based on the validation loss.
  * The network trained for 200 epochs and was saved every 10 epochs if the current validation loo decreased when compared to the previous validation loss.

* VAE Perception Demo:
  Left image shows the real observation, right image shows the VAE reconstructed image <br>
  <img src="https://user-images.githubusercontent.com/43849409/146668563-1b215f51-c91e-44ad-ba0b-91658bd0dab9.gif" width="250" height="200"/> 

* Based on the trained VAE, a new dataset consisting of latent vectors and the corresponding data label can be generated. The latent representation seems to make the original data linearly separable in the embedded space(fig. below) and can be visualized using tensorboard and T-SNE and UMAP plots. 
  * T-SNE projection <br>
    <img src="https://user-images.githubusercontent.com/43849409/148103979-a48d94b4-70b7-4203-be2d-f94eb6207c4b.png" width="250" height="225"/>

  * UMAP projection <br>
    <img src="https://user-images.githubusercontent.com/43849409/148104019-4e0c14ca-3e4a-4522-806f-08dfd3c3df5f.png" width="250" height="225"/>

* A linear classifier (logistic regression) is chosen to fit the embedding space data and the classification report is shown below <br>
  <img src="https://user-images.githubusercontent.com/43849409/148104063-fe647193-0b0a-4f6b-816c-686361f9b0c6.png" width="450" height="400"/>


## Workflow 2 (Topological Data Analysis):
* From the Confusion Matrix above, it’s visible that some of the misclassifications caused are between the following letters:
  1. A with H
  2. B with D and E
  3. C with E and G
  4. I with J

* All these letters structurally are similar, in the sense that if we were to attach the top ends of H to form a loop it’d look like an A. Although they are structurally similar, they are topologically very different. Topological Data Analysis(TDA) helps us in understanding different shapes present in the data. They are powerful tools to both visualize and aid machine learning algorithms in achieving a certain task. For example, let’s look at one of the topological characteristics (in this case the number of loops present in the structure) of the class labels:
  1. A has 1 loop
  2. B has 2 loops
  3. C has 0 loops
  4. D has 1 loop
  5. E has 0 loops
  6. F has 0 loops
  7. G has 0 loops
  8. H has 0 loops
  9. I has 0 loops
  10. J has 0 loops

* From the topological characteristics, if we were to find a way to calculate the number of loops,  then we can predict the letter B with lesser misclassifications as it’s the only label with 2 loops. The purpose of this experiment is to understand how we can use such topological features that can aid in classification. The following topics will be discussed briefly with examples to comprehend and compute relevant topological features:
  1. Filtration
  2. Simplicial Complex and Persistent Homology
  3. Feature Representation
  4. TDA Feature Generation Pipeline

### Filtration:
* Filtration is an important step in any TDA pipeline as it’s the first step to create something called Complex. Complex is a way of forming a topological space from distances in a set of points. Distances between points are defined by something called the Metric and points closer than a specified threshold are connected by an edge in a complex. Construction of a complex is necessary to understand the global structure of the data.

* One such example of a filtration for images is the Radial filtration. It assigns each pixel of a binary image(grayscale) a value that’s computed from a reference pixel, called the “center”, and of a “radius”. If the binary pixel is active(non-zero) and lies within the ball defined by this reference center and radius, then the assigned value equals its distance from the reference pixel. For an inactive pixel, the assigned value equals the maximum distance between any pixel of the image and the reference center pixel, plus one. Lets visualize the Radial filtration:

* Binarized Image: <br>
  <img src="https://user-images.githubusercontent.com/43849409/148111530-0b254595-ce09-42e7-a3a5-2068130a02c4.png" width="250" height="200"/>


* Radial Filtration (center is the pixel at upper right corner): <br>
  <img src="https://user-images.githubusercontent.com/43849409/148111553-84cfef6b-3749-4f99-8f25-530312b7d8b2.png" width="250" height="200"/>


### Simplicial Complex and Persistent Homology:
* In mathematics, a simplicial complex is a set composed of points, line segments,triangles, and their n-dimensional counterparts (see illustration). A simplex is defined as the fundamental entity in a simplicial complex. Eg. 0-simplex is a point, 1-simplex is an edge, 2-simplex is a triangle and so on. They are formed as a result of edge formation, as described above, between points that lie within a certain ball of radius determined by the distance threshold (fig. below). 

* Simplicial Complex: <br>
  <img src="https://user-images.githubusercontent.com/43849409/148111564-41361fe3-7a5a-459f-9cb9-091c3fa870ff.png" width="225" height="200"/>

* Homology is a rigorous mathematical method for defining and categorizing holes in a manifold formed by the simplicial complex. Persistent Homology can be considered as a historical ledger of different homology formations when the distance threshold (mentioned above) varies from 0 to its maximum value.

* Persistent Homology: <br>
  <img src="https://user-images.githubusercontent.com/43849409/148112406-32b0a941-f532-4ab6-b44d-fe5e8542799d.png" width="450" height="300"/>


* The relevance of Persistent Homology to the current exercise is that it helps to compute different topological features like loops. As seen above, the letter B has two loops and the Persistence Diagram should reflect this. We can compute the Persistent Homology by forming a cubical simplicial complex from the filtration that we computed before and then plotting the historical events of hole formations using Homology (fig. below)

* Persistence Diagram for B: <br>
  <img src="https://user-images.githubusercontent.com/43849409/148112430-7d1820b6-f5a1-42e9-bdbe-48d64ff932c2.png" width="300" height="250"/>

  H0 - Edges (Red Dots) <br> 
  H1 - Loops (Green Dots) <br>
  
* As shown in the Persistence Diagram, the filtration for image B has two loops and three edges. Hence different topological information can be captured for different images, Persistence Diagram for D is shown below and as expected it captures the single D-loop in H1 (fig. below)

  <img src="https://user-images.githubusercontent.com/43849409/148112442-e442e0da-ba27-4adf-909e-ae7e60117d39.png" width="300" height="250"/>



### Feature Representation:
* Persistent Homology needs to be translated to a meaningful metric for it to be considered as a feature. One of the ways this can be achieved is by convolving a gaussian kernel on the persistence diagram, a procedure achieved via the heat kernel. For each of the Homology dimensions, the Heat Kernels amplitude is computed to get the vectorized feature. 

* Heat Kernel Maps for H0 and H1 of B: <br>
  <img src="https://user-images.githubusercontent.com/43849409/148112465-ca74ae0d-77f9-46d5-bcff-1aa3feae1db1.png" width="500" height="500"/>

### TDA Feature Generation Pipeline:
* Giotta-tda python library is a high performance TDA library that helps us compute Persistent Homology features and it’s compatible with the Sklearn Pipeline method. There’s a whole bunch of feature vectorization that can be done and for this exercise the following features were computed resulting in 476 TDA based features:
    1. Amplitudes using the following metrics:
      a. Bottleneck distances
      b. Wasserstein distances
      c. Persistence Landscapes
      d. Betti vectors
      e. Heat-Kernels
    2. Persistence Entropy

### Logistic Regression using TDA Features:
* Before combining these features with the VAE features, we can inspect the predictive power of TDA features by modeling it using a logistic regression model and checking the feature importance across all classes (fig below.).
 
  <img src="https://user-images.githubusercontent.com/43849409/148112502-3cc1b60c-b7a5-46ad-985d-c408f4c618b7.png" width="500" height="300"/>

* From the above plot, we can see that there’s a good cohort of features that have predictive power. A logistic regression model was trained based on these TDA features to check if they can provide any potential boost and the confusion-matrix is as follows:


* Confusion Matrix: <br>
  <img src="https://user-images.githubusercontent.com/43849409/148112537-f4192dd1-240e-4cca-b530-b95c13f753eb.png" width="450" height="400"/>


## Ensemble Models
* The summary of exercises done so far are as follows:
    * Modelled a Variational Autoencoder to get the latent-space representation of the notMNIST dataset.
    * Modelled a Logistic Regression Classifier using the VAE features (also called latent space features) and produced a classification accuracy of ~88.9%
    * Implemented a pipeline to generate TDA based Persistent Homology features using Giotta-tda library. 
    * Modelled a Logistic Regression Classifier using the TDA features to check for predictive power of the TDA features and produced a classification accuracy of ~84.4%

* Not all features generated by the TDA pipeline seemed useful and there was never an effort to check how they’d interact when combined with the VAE features. The next step in the project is to fetch a good cohort of these features for the best predictive performance and there are multiple ways to approach this problem. In this project I have decided to automate this step by using a tree-based ensemble model called Gradient Boosted Decision Tree (GBDT). 

* The GBDT model inherently checks for the best features to create multiple simple decision trees based on varying subsets of data. The GBDT model then combines these simple trees based on their respective cross-validation accuracy to form its prediction strategy which results in a strong classifier with low variance and low bias. As discussed before, the features from TDA methods and VAE were concatenated and the resulting dataset shape turned out to have 508 features and 14k rows. GBDT is a resource intensive algorithm and a new high performance library called LightGBM (By Microsoft) was used to implement it. The final model that was trained had the following hyper-parameters and gave a classification accuracy of ~94%:

     <img src="https://user-images.githubusercontent.com/43849409/148112591-d9f5feee-0b6e-4155-9970-096883953522.png" width="400" height="75"/>

* Confusion Matrix: <br>
  <img src="https://user-images.githubusercontent.com/43849409/148112615-87176d52-52d0-4ca8-8944-a2fe6e9efb94.png" width="450" height="400"/>



## Confusion Matrix Analysis
* As seen from the confusion matrix of the GBDT classifier, misclassifications have reduced drastically for almost all of the class labels. We can visualize the percentage improvement by checking the trace of difference of confusion matrices between the two methods. The resultant percentage heatmap is as follows:

  <img src="https://user-images.githubusercontent.com/43849409/148455088-dbf1a9ff-e79f-48f3-94f6-058bb15bc324.png" width="450" height="400"/>

* Almost every class had 3% and above improvement with the top-4 being:
    1. E (Became robust to misclassification with class C and I)
    2. B (Became robust to misclassification with class D and E)
    3. G (Became robust to misclassification with class E, F, H and I)
    4. A (Became robust to misclassification with class H and J)

## Conclusion
* Topological Data Analysis is a powerful tool in identifying global structures of the data. The features generated from TDA are highly robust to noise. When combined with Deep-Learning approaches, a powerful class of feature engineering algorithms evolve. The best part about the whole exercise is that simple and linear models can perform like state-of-the-art models with more interpretability and explainability just by providing the right set of features.

## References
1. https://arxiv.org/pdf/1909.10604.pdf
2. https://arxiv.org/pdf/1904.07403.pdf
3. https://arxiv.org/pdf/1906.00722.pdf
4. https://arxiv.org/pdf/1910.12939.pdf
5. https://lightgbm.readthedocs.io/en/latest/
6. https://giotto-ai.github.io/gtda-docs/0.5.1/library.html
