import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,f1_score,recall_score, precision_score

def plot_histories(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(10,8))
    plt.subplot(1,2,1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('acc')

    plt.subplot(1,2,1)
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('loss')
    plt.show()

def evaluations(y_test, pred_labels):
    acc = accuracy_score(y_test,pred_labels)
    prec = precision_score(y_test,pred_labels)
    f1 = f1_score(y_test,pred_labels)
    recall = recall_score(y_test,pred_labels)
    print(f'accuracy: {acc:.2f}, precision: {prec:.2f}, f1 score: {f1:.2f}, recall: {recall:.2f}')