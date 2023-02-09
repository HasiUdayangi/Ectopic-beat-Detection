import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix


plt.title('Accuracies vs Epochs')
plt.plot(model_history[0].history['accuracy'], label='Training Fold 1')
plt.plot(model_history[1].history['accuracy'], label='Training Fold 2')
plt.plot(model_history[2].history['accuracy'], label='Training Fold 3')
#plt.plot(model_history[3].history['accuracy'], label='Training Fold 4')
#plt.plot(model_history[4].history['accuracy'], label='Training Fold 5')
plt.legend()
plt.show()



plt.figure(figsize=(10,5))
plt.title('Train Accuracy vs Val Accuracy')
plt.plot(model_history[0].history['accuracy'], label='Train Acc Fold 1', color='black')
plt.plot(model_history[0].history['val_accuracy'], label='Val Acc Fold 1', color='black', linestyle = "dashdot")
plt.plot(model_history[1].history['accuracy'], label='Train Acc Fold 2', color='red', )#
plt.plot(model_history[1].history['val_accuracy'], label='Val Acc Fold 2', color='red', linestyle = "dashdot")
plt.plot(model_history[2].history['accuracy'], label='Train Acc Fold 3', color='green', )
plt.plot(model_history[2].history['val_accuracy'], label='Val Acc Fold 3', color='green', linestyle = "dashdot")
#plt.plot(model_history[3].history['accuracy'], label='Train Acc Fold 4', color='blue', )
#plt.plot(model_history[3].history['val_accuracy'], label='Val Acc Fold 4', color='blue', linestyle = "dashdot")
#plt.plot(model_history[4].history['accuracy'], label='Train Acc Fold 5', color='orange', )
#plt.plot(model_history[4].history['val_accuracy'], label='Val Acc Fold 5', color='orange', linestyle = "dashdot")
plt.legend()
plt.show()


model_fromALL = load_model('final_model.h5')


#combination of test data sets
pred_test = model_fromALL.predict(X_test)
pred_test = np.argmax(pred_test, axis=-1)

xx1 = confusion_matrix(y_test, pred_test)
tks1 = ['NOTEB','VEB', 'SVEB']

plot_conf_matrix(xx1, tks1)


print('Classification Report')
print(classification_report(y_test, pred_test, target_names=['NOTEB','VEB', 'SVEB']))


total = sum(sum(xx1))
#####from confusion matrix calculate accuracy
accuracy = (xx1[0,0] + xx1[1,1]) / total
print ('Accuracy : ', accuracy)

sensitivity = xx1[0,0] / (xx1[0,0] + xx1[0,1])
print('Sensitivity : ', sensitivity)

specificity = xx1[1,1] / (xx1[1,0] + xx1[1,1])
print('Specificity : ', specificity)

precision = xx1[0,0] / (xx1[0,0] + xx1[1,0])
print('Precision : ', precision)

nepro = xx1[1,1] / (xx1[1,1] + xx1[0,1])
print('Negative procuctivity : ', nepro)


from sklearn.metrics import precision_recall_fscore_support

label = [0,1,2]
result = []
for label in label:
    precision, recall, f_score, support = precision_recall_fscore_support(np.array(y_test) == label, np.array(pred_test)==label)
    result.append([label, recall[0], recall[1], precision[1], f_score[1]])
df=pd.DataFrame(result, columns=["label","specificity",'recall','precision','f_score'])
print(df)

multiclass_roc_auc_score(y_test, pred_test, savename='model_ALL.png')