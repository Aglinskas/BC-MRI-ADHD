# BC-MRI-ADHD
ADHD CVAE 


# Folder structure
 - Assets: For outside data like CSV files, brain masks, etc.
 - Code: Where code lives: .py .ipynb files etc
 - Data: Derivatives of input data (smaller than 100mb): e.g. arrays of brain data
 - Results: Outputs of analyses: arrays of averages, data for  figures
 - Figures: jpgs, pdfs of figures (use plt.save_fig(fn='bla'))

# Requirements
 [ANTSpy](https://github.com/ANTsX/ANTsPy)
 Tensorflow (`import tensorflow as tf`) or [instal tf](https://www.tensorflow.org/install)