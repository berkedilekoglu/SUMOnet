
from sklearn import metrics

def f1_score(y_true, y_probs):

    return metrics.f1_score(y_true, y_probs.argmax(-1))

def mcc(y_true, y_probs):

    return metrics.matthews_corrcoef(y_true, y_probs.argmax(-1))

def rocAuc_score(y_true, y_probs):

    lr_fpr, lr_tpr, _ = metrics.roc_curve(y_true, y_probs[:,1])

    return metrics.auc(lr_fpr, lr_tpr)

def aupr_score(y_true, y_probs):

    precision, recall, _ = metrics.precision_recall_curve(y_true, y_probs[:,1])
    
    return metrics.auc(recall, precision)

def evaluate(y_true, y_probs, metrics_):
    
    """
    Args:

        y_true: gold labels -> It is a 2-dim array: [[1,0],[0,1],[1,0],[1,0]]
        y_probs: predicted probabilities -> It is a 2-dim array: [[0.4,0.6],[0.2,0.8],[0.7,0.3],[0.6,0.4]]
        metrics: evaluation metrics -> can be a list of strings ['f1', 'mcc', 'roc', 'aupr'] or a string 'roc'

    Output:

        outputEval: a dictionary that contains evaluation metric name and its value -> {'f1':0.72, 'mcc': 0.42}
    """


    metricsDict = {'f1':f1_score, 'mcc':mcc, 'roc':rocAuc_score, 'aupr':aupr_score}
    outputEval = {}

    if type(metrics_) == str:

        if metrics_.lower() not in metricsDict:

                raise ValueError(f"{metrics_.lower()} is not supported! f1, mcc, roc and aupr are supported evaluation metrics")

        else:

            outputEval[metrics_.lower()]  = metricsDict[metrics_.lower()](y_true, y_probs)

        return outputEval

    elif type(metrics_) == list:

        metricNames = [x.lower() for x in metrics_]

        for metricName in metricNames:

            if metricName not in metricsDict:

                raise ValueError(f"{metricName} is not supported! f1, mcc, roc and aupr are supported evaluation metrics")

            else:

                outputEval[metricName]  = metricsDict[metricName](y_true, y_probs)

        return outputEval

    else:

        raise TypeError("metrics must be a string or list of strings")

