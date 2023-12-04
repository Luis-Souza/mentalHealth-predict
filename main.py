import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from preprocessing import Processing
from normalizadeModel import NormalizedModel

from utils import evaluations, plot_histories

# read the file
df = pd.read_csv("./file/student_mental_health.csv")

# redefine columns
new_columns = ['Time','Gender','Age','Major','Year','CGPA','Marriage','Depression','Anxiety','Panic','Treatment']
df.columns = new_columns

df.dropna(inplace=True)

processing = Processing(df)
df = processing.passResponseTextsToNumbers()
df = processing.passYersToNumber()
df = processing.putWeightsCGPA()
df = processing.defineNumeberToRepresentationSex()
df = processing.encodeLabels()
df = processing.dropColumn('Time')

norm_X_train, X_train, y_train, norm_X_test, y_test, X_test = processing.trainProcess()

normalizedModel = NormalizedModel(norm_X_train=norm_X_train, X_train=X_train, y_train=y_train, norm_X_test=norm_X_test, y_test=y_test, X_test=X_test)
plot_histories(normalizedModel.history())
print(normalizedModel.evaluate())

pred_labels = normalizedModel.prediction()

evaluations(y_test, pred_labels)
print(classification_report(y_test, pred_labels))

plt.figure(figsize=(18,4))
df.corr()['Depression'].sort_values(ascending = True)[:-1].plot(kind='bar')
plt.xlabel('Features')
plt.ylabel('Correlation to blood pressure categories')
plt.title('Features\' correlation to the Depression')
