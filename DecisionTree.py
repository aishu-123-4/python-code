import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from sklearn.model_selection import learning_curve, StratifiedKFold,
train_test_split
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
data = np.c_[cancer.data, cancer.target]
columns = np.append(cancer.feature_names, ["target"])
sizeMeasurements =pd.DataFrame(data, columns =columns)
X= sizeMeasurements[sizeMeasurements.columns[:-1]]
y= sizeMeasurements.target
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size
=0.2)
sns.set_style("whitegrid")
plotOne = sns.FacetGrid(sizeMeasurements, hue ="target",aspect= 2.5)
plotOne.map(sns.kdeplot,'mean area',shade =True)
plotOne.set (xlim =(0, sizeMeasurements['mean area'].max()))
plotOne.add_legend()
plotOne.set_axis_labels('mean area' , 'Proportion' )
plotOne.fig.suptitle ('Area vs Diagnosis (Blue = Malignant ; Orange =
Benign)')
plt.show()
sns.set_style("whitegrid")
plotTwo = sns.FacetGrid(sizeMeasurements, hue ="target",aspect= 2.5)
plotTwo.map(sns.kdeplot,'mean concave points',shade =True)
plotTwo.set (xlim =(0, sizeMeasurements['mean concave points'].max()))
plotTwo.add_legend()
plotTwo.set_axis_labels('mean concave points' , 'Proportion' )
plotTwo.fig.suptitle ('# of Concave Points vs Diagnosis (Blue = Malignant
; Orange = Benign)')
plt.show()
print("\n Feature Correlation :\n")
g= sns.heatmap(X_train.corr(),cmap="BrBG",annot=False)
sizeMeasurements2 = sizeMeasurements.drop(['mean radius','mean
perimeter','mean smoothness' , 'mean compactness','mean concavity','mean
concave points','mean fractal dimension','radius error','texture
error','perimeter error','area error','smoothness error','compactness
error','concavity error','concave points error','symmetry error','fractal
dimension error','worst radius','worst perimeter','worst
smoothness','worst compactness','worst concave points','worst
symmetry','worst fractal dimension','worst texture','worst area','worst
concavity'],axis =1)
X2 = sizeMeasurements2[sizeMeasurements2.columns[:-1]]
y2=sizeMeasurements2.target
X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, y2, test_size
=0.2)
print('\nFeature Names: \n\n',X2.columns.values,"\n")
print("\nFeature Correlation:\n")
#g=sns.heatmap(X_train2.corr(),cmpa="BrBG",annot=False)
X3=X
y3=y
variance_pct = 5
pca = PCA(n_components=variance_pct)
X_transformed = pca.fit_transform(X3,y3)
X3pca = pd.DataFrame(X_transformed) # Create a data frame from the PCA'd
data
X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3, y3,
test_size=0.2)
print('\n Feature Names: \n\n', X3pca.columns.values, "\n")
print("\nFeature Correlation: \n")
g = sns.heatmap(X_train3.corr(),cmap="BrBG",annot=False)
def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion
matrix',cmap=plt.cm.Blues):
plt.imshow(cm, interpolation='nearest', cmap=cmap)
plt.title(title)
plt.colorbar()
tick_marks = np.arange(len(classes) )
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
plt.text(j, i, format(cm[i, j],
fmt),horizontalalignment="center",color="white" if cm[i, j]> thresh else
"black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel( 'Predicted label')
dict_characters = {0: 'Malignant', 1: 'Benign'}
(X1, y1) = load_breast_cancer(return_X_y = True)
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X1,y1,random_state=0)
clf = RandomForestClassifier(max_features=8,random_state=0)
clf.fit(X_train1,Y_train1)
print('Accuracy of Random Forest Classifier on training data:
{:.2f}'.format(clf.score(X_train1,Y_train1)))
print('Accuracy of Random Forest Classifier on testing data:
{:.2f}'.format(clf.score(X_test1,Y_test1)))
model = clf
prediction = model.predict(X_test1)
cnf_matrix = confusion_matrix(Y_test1, prediction)
plt.show()
plot_confusion_matrix(cnf_matrix, classes=dict_characters,title='Confusion
matrix')
plt.show()





