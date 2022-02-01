import os
import cv2
import numpy as np
import seaborn as sns
from tqdm import tqdm
from PIL import Image as PIL_Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


import torch as T
from torch.utils.data import TensorDataset, DataLoader


def get_dataloaders(data_dict, name=None, batch_size=64, return_label=False):
    """
    Utility function to create data-loader for notMNIST dataset
    :param data_dict:
    :param name:
    :param batch_size:
    :param return_label:
    :return:
    """
    x, y = [], []
    skip_dict = {}
    for data_class, paths in tqdm(data_dict.items()):
        for path in paths:
            try:
                img_arr = np.expand_dims(np.asarray(PIL_Image.open(path)), axis=0)
                label = ord(data_class) - ord('A')
                x.append(img_arr)
                y.append(label)
            except Exception as e:
                if data_class in skip_dict:
                    skip_dict[data_class] += 1
                else:
                    skip_dict[data_class] = 1

    print('Missed examples -> ', skip_dict)
    print(name + ' size: {}'.format(len(x)))
    x = T.from_numpy(np.flip(np.array(x), axis=0).copy()).type('torch.FloatTensor')
    y = x if not return_label else T.from_numpy(np.flip(np.array(y), axis=0).copy()).type('torch.LongTensor')
    return DataLoader(TensorDataset(x, y), batch_size=batch_size)


def get_video(test, pred, name):
    size = (28 * 2, 28)
    video_fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    video = cv2.VideoWriter(name + '.avi', video_fourcc, 1, size)
    print('Converting predictions to video')
    for i in tqdm(range(test.shape[0])):
        arr1 = np.einsum("chw -> hwc", test[i, :, :, :])
        arr2 = np.einsum("chw -> hwc", pred[i, :, :, :])
        arr1 = np.asarray(arr1, dtype=np.uint8)
        arr2 = np.asarray(arr2, dtype=np.uint8)
        arr = cv2.hconcat([arr1, arr2])
        video.write(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))

    video.release()


def get_clf_stats(clf, x, y):
    y_pred = clf.predict(x)
    y_true = y
    print('Classifier Accuracy: ', accuracy_score(y_true, y_pred))
    target_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    print('Classification Report: \n', classification_report(y_true, y_pred, target_names=target_names))
    ax = plt.subplot()
    cf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(cf_matrix, annot=True, fmt='g', cmap='plasma_r')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(target_names)
    ax.yaxis.set_ticklabels(target_names)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.show()
    return cf_matrix


def save_data(train_arr, test_arr, name):
    cur_dir = os.getcwd() + '/processed_data'
    np.save(cur_dir + '/' + name + '_train', train_arr)
    np.save(cur_dir + '/' + name + '_test', test_arr)


def load_data(name):
    cur_dir = os.getcwd() + '/processed_data'
    train_arr = np.load(cur_dir + '/' + name)
    test_arr = np.load(cur_dir + '/' + name)
    return train_arr, test_arr
