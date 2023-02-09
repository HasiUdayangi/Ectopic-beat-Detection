from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn import metrics


def evaluate_model(history,X_test,y_test,model):
    scores = model.evaluate((X_test),y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    
    print(history)
    fig1, ax_acc = plt.subplots()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model - Accuracy')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.show()
    
    fig2, ax_loss = plt.subplots()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model- Loss')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()
    target_names=['0','1','2','3','4']
    
    y_true=[]
    for element in y_test:
        y_true.append(np.argmax(element))
    prediction_proba=model.predict(X_test)
    prediction=np.argmax(prediction_proba,axis=1)
    cnf_matrix = confusion_matrix(y_true, prediction)
    

    
def plot_conf_matrix(conf_matrix, ticks, title: str = None, xlabel: str = None, ylabel: str = None, savename: str = None):

    if title is None:
        title = 'Confusion Matrix'
    
    if xlabel is None:
        xlabel = 'CNN Model'
    
    if ylabel is None:
        ylabel = 'label'    
    
    ax = plt.subplot()
    sns.heatmap(conf_matrix, annot=True, ax = ax, fmt="d"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_ticklabels(ticks)
    ax.yaxis.set_ticklabels(ticks)
    if savename is not None:
        plt.savefig(savename)
    else:
        plt.show()
        

def multiclass_roc_auc_score(y_test, y_pred, savename, classes = ['NOTEB','VEB', 'SVEB'], average="macro"):
    fig, c_ax = plt.subplots(1,1, figsize = (12, 8))
    lb = preprocessing.LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(classes): 
        fpr, tpr, thresholds = metrics.roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, metrics.auc(fpr, tpr)))
        
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    plt.title('Multiclass ROC')
    plt.legend()
    plt.savefig(savename)
    return metrics.roc_auc_score(y_test, y_pred, average=average)