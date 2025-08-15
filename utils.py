from sklearn.metrics import confusion_matrix
from numpy import ndarray
import cv2 as cv
import torch
import numpy as np

def dice(TP, FP, FN):
   return (2*TP)/(FP + (2*TP) + FN)


def iou(TP, FP, FN):    
    return TP/(TP + FP + FN)


def ppv(TP, FP):
    return TP/(FP + TP)


def accuracy(TP, TN, FP, FN):
    return (TP + TN)/(TP + TN + FP + FN)


def sensitivity(TP, FN):
    return TP/(TP+FN)

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i]==y_pred[i]==1:
           TP += 1
        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:
           FP += 1
        if y_actual[i]==y_pred[i]==0:
           TN += 1
        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:
           FN += 1

    return (TN, FP, FN, TP)





true = cv.imread("/content/brisc2025_train_00028_gl_ax_t1.png", cv.IMREAD_GRAYSCALE)
true_mask = true.flatten() # Flatten the true mask

# Binarize the predicted mask
pred_mask = mask.squeeze(0).cpu().detach().numpy() # Remove the batch dimension

# Apply the binarization logic
pred_mask[pred_mask < 0] = 0
pred_mask[pred_mask > 0] = 1
fig, ax = plt.subplots(1,2)
_ = pred_mask.squeeze(0)
ax[0].imshow(true, cmap='gray')
ax[0].set_title("Original Mask")
ax[1].imshow(_, cmap='gray')
ax[1].set_title("Predicted Mask")

pred_mask = pred_mask.flatten().astype(int) # Flatten and convert to integer type
pred_mask = np.uint8(pred_mask)

print(true_mask.shape)
print(pred_mask.shape)

true_mask = np.array(true_mask)
pred_mask = np.array(pred_mask)
# Confusion matrix of the masks

print(true_mask)
print(pred_mask)

print(np.dtype(true_mask[0]))
print(np.dtype(pred_mask[0]))
TN, FP, FN, TP = perf_measure(true_mask, pred_mask)




print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")