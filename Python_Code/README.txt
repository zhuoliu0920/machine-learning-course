In this folder, I replicated all the graphs from reports for this course by using matplotlib in Python. I originally got all the plots in R, but it is a good practice by using Python instead.

script1.py: Get the bar plot for the counts of each classes, and histograms for each predictor on each class.

script2.py: By applying PCA, project original data space onto first and second 2 principle component spaces. Then, applying LDA on the PC space, project onto first and second linear discriminant spaces.

script3.py: Similar to script2, but using kernal PCA instead. Here, gaussian (rbf) kernal is used.

script4.py: Build classifiers by SVM with different kernals (linear, gaussian and polynomial). Find the optimal parameters.
