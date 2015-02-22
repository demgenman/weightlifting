# weightlifting
Weight lifting excercise classification

Files in this repository:
- README.md: This file.
- wle.Rmd: R markdown file. Knitting to html takes about 10 minutes on a Dell E6430 Windows 7 64 bit.
- wle.md: Markdown html file. Images are in the figure folder.
- wle.html: The knitted html file.
- wle.pdf: The knitted pdf file.
- figure/diagnostic_model-1.png: Variable importance plot, initial model.
- figure/diagnostic_mode3-1.png: Variable importance plot, initial model, y-axis = MDG, x-axis = MDA.
- figure/diagnostic_mode5-1.png: Variable importance plot, final model.
- figure/diagnostic_mode6-1.png: Histogram of prediction margin.
- figure/diagnostic_mode7-1.png: ROC curve panel plot for the five response classes.

The code in wle.Rmd assumes that the following files have been downloaded to the the working directory:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv: the training data set.
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv: the dataset with the 20 observations to predict.

