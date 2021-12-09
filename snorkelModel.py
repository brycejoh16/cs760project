# --- Imports ------------------------------------------------------------------
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LFApplier
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
import labellingFunctions as lf
import loadData

# --- Snorkel Labeling Model ---------------------------------------------------
def getLabelMatrix():
    # labeling functions list
    lfs = [
        lf.classifyEdgeDetectRatio,
        lf.classifyEdgeDetectHorizontal,
        lf.classifyEdgeDetectVertical,
        lf.classifyRatioPeakCount,
        lf.classifyVerticalPeakCount,
        lf.classifyHorizontalPeakCount,
        lf.classifyL2Norm,
        lf.classifyFillSum,
        lf.classifyFillCount,
        lf.classifyPixelCount,
    ]

    # create L from data; matrix of labeling functions and their classifications
    applier = LFApplier(lfs=lfs)
    labelMatrix = applier.apply(data)
    return labelMatrix, lfs

def getMajorityPredictions(labelMatrix):
    # majority model
    majority_model = MajorityLabelVoter()
    majorityPreds = majority_model.predict(L=labelMatrix)
    return majorityPreds

def getModelPredictions(labelMatrix):
    # Snorkel labeling model
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=labelMatrix, n_epochs=500, log_freq=100, seed=123)
    modelPreds = label_model.predict(L=labelMatrix)
    return modelPreds

def printAccuracy(predictions, targets, modelName):
    correct = 0
    for p, t in zip(predictions, targets):
        if p == t:
            correct += 1

    percent = (correct/len(targets))*100
    print("Correctly classified by the ", modelName, ": ", percent)

if __name__ == "__main__":
    data, targets = loadData.loadMNIST()
    labelMatrix, lfs = getLabelMatrix()
    print(LFAnalysis(L=labelMatrix, lfs=lfs).lf_summary())
    majorityPreds = getMajorityPredictions(labelMatrix)
    modelPreds = getModelPredictions(labelMatrix)
    printAccuracy(majorityPreds, targets, "Majority Model")
    printAccuracy(modelPreds, targets, "Label Model")
