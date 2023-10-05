import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from quantifiers.ClassifyCountCorrect.AdjustedClassifyCount import AdjustedClassifyCount
from quantifiers.ClassifyCountCorrect.ClassifyCount import ClassifyCount
from quantifiers.ClassifyCountCorrect.MAX import MAX
from quantifiers.ClassifyCountCorrect.MedianSweep import MedianSweep
from quantifiers.ClassifyCountCorrect.ProbabilisticAdjustedClassifyCount import ProbabilisticAdjustedClassifyCount
from quantifiers.ClassifyCountCorrect.ProbabilisticClassifyCount import ProbabilisticClassifyCount
from quantifiers.ClassifyCountCorrect.T50 import T50
from quantifiers.ClassifyCountCorrect.X import X
from quantifiers.DistributionMatching.DyS import DyS
from quantifiers.DistributionMatching.HDy import HDy
from quantifiers.DistributionMatching.SORD import SORD


def apply_quantifier(quantifier, clf, thr, measure, train_test, test_sample):
    X_train = train_test[0]
    y_train = train_test[2]

    if quantifier == "CC":
        cc = ClassifyCount(classifier=clf, threshold=thr)
        cc.fit(X_train, y_train)

        return cc.predict(test_sample)

    if quantifier == "ACC":
        acc = AdjustedClassifyCount(classifier=clf, threshold=thr)
        acc.fit(X_train, y_train)

        return acc.predict(test_sample)

    if quantifier == "PCC":
        pcc = ProbabilisticClassifyCount(classifier=clf)
        pcc.fit(X_train, y_train)

        return pcc.predict(test_sample)

    if quantifier == "PACC":
        pacc = ProbabilisticAdjustedClassifyCount(classifier=clf, threshold=thr)
        pacc.fit(X_train, y_train)

        return pacc.predict(test_sample)

    if quantifier == "X":
        x_qtf = X(classifier=clf)
        x_qtf.fit(X_train, y_train)

        return x_qtf.predict(test_sample)

    if quantifier == "MAX":
        max_qtf = MAX(classifier=clf)
        max_qtf.fit(X_train, y_train)

        return max_qtf.predict(test_sample)

    if quantifier == "T50":
        t50 = T50(classifier=clf)
        t50.fit(X_train, y_train)

        return t50.predict(test_sample)

    if quantifier == "MS":
        ms = MedianSweep(classifier=clf)
        ms.fit(X_train, y_train)

        return ms.predict(test_sample)

    if quantifier == "HDy":
        hdy = HDy(classifier=clf)
        hdy.fit(X_train, y_train)

        return hdy.predict(test_sample)

    if quantifier == "DyS":
        dys = DyS(classifier=clf, similarity_measure=measure)
        dys.fit(X_train, y_train)

        return dys.predict(test_sample)

    if quantifier == "SORD":
        sord = SORD(classifier=clf)
        sord.fit(X_train, y_train)

        return sord.predict(test_sample)

def experiment(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
    train_test = [X_train, X_test, y_train, y_test]

    clf = RandomForestClassifier(n_estimators=200)

    # Experimental setup
    counters = ["CC", "ACC", "PCC", "PACC", "X", "MAX", "T50", "MS", "HDy", "DyS", "SORD"]
    measure = "topsoe"  # Default measure for DyS

    # How many replicates it will take
    niterations = 3

    # Test sizes
    batch_sizes = [100]

    # Positive class proportion
    alpha_values = [round(x, 2) for x in np.linspace(0, 1, 11)]  # class proportion

    df_test = pd.concat([X_test, y_test], axis='columns')

    df_test_pos = df_test.loc[df_test['class'] == 1]  # seperating positive test examples
    df_test_neg = df_test.loc[df_test['class'] == 0]  # seperating negative test examples

    columns = ["sample", "Test_size", "alpha", "actual_prop", "pred_prop", "abs_error", "quantifier"]
    table = pd.DataFrame(columns=columns)
    print(table)
    for sample_size in batch_sizes:  # Varying test set sizes
        for alpha in alpha_values:   # Varying positive class distribution
            for iteration in range(niterations):
                pos_size = int(round(sample_size * alpha, 2))
                neg_size = sample_size - pos_size

                if pos_size is not sample_size:
                    sample_test_pos = df_test_pos.sample(int(pos_size), replace=False)
                else:
                    sample_test_pos = df_test_pos.sample(frac=1, replace=False)

                sample_test_neg = df_test_neg.sample(int(neg_size), replace=False)

                sample_test = pd.concat([sample_test_pos, sample_test_neg])
                test_label = sample_test["class"]

                test_sample = sample_test.drop(['class'], axis='columns')

                n_pos_sample_test = list(test_label).count(1)                       # Counting num of actual positives in test sample
                calcultd_pos_prop = round(n_pos_sample_test / len(sample_test), 2)  # actual pos class prevalence in generated sample

                for co in counters:
                    quantifier = co

                    # .............Calling of Methods..................................................
                    pred_pos_prop = apply_quantifier(quantifier=quantifier, clf=clf, thr=0.5, measure=measure,
                                                     train_test=train_test, test_sample=test_sample)

                    pred_pos_prop = round(pred_pos_prop[1], 2) # Getting only the positive proportion

                    # ..............................RESULTS Evaluation.....................................
                    abs_error = round(abs(calcultd_pos_prop - pred_pos_prop), 2)  # absolute error
                    result = {'sample': iteration+1, 'Test_size': sample_size, 'alpha':alpha,
                              'actual_prop': calcultd_pos_prop, 'pred_prop': pred_pos_prop,
                              'abs_error': abs_error, 'quantifier': quantifier}
                    result = pd.DataFrame([result])

                    table = pd.concat([table, result], ignore_index=True)
                    print(table)

    return table


if __name__ == '__main__':
    dts_data = pd.read_csv('C:\\1Faculdade\\EstagioSupervisionado\\AedesQuinx.csv')
    dts_data.rename(columns={'species': 'class'}, inplace=True)

    mapping = {'AA': 1, 'CQ': 0}
    dts_data['class'] = dts_data['class'].map(mapping)
    dts_data = dts_data[dts_data['temp_range'] == 4]

    size = 500
    pos_class = dts_data[dts_data['class'] == 1]
    neg_class = dts_data[dts_data['class'] == 0]

    pos_class = pos_class.sample(n=size)
    neg_class = neg_class.sample(n=size)

    dts_data = pd.concat([pos_class, neg_class], ignore_index=True)

    print(dts_data)

    label = dts_data['class']
    dataset = dts_data.drop(['class'], axis='columns')

    # print(dataset)
    # print(label)

    result_table = experiment(dataset, label)

    print(result_table)
    path = 'C:\\1Faculdade\\EstagioSupervisionado\\result_table2.csv'

    result_table.to_csv(path, index=False)

