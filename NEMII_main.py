# coding=UTF-8
import gc
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier
import random
from datetime import datetime
from SDNE import graph, sdne

#将matrix变成edgelist格式
def Net2edgelist(miRNA_disease_matrix_net):
    none_zero_position = np.where(np.triu(miRNA_disease_matrix_net) != 0)
    none_zero_row_index = np.mat(none_zero_position[0],dtype=int).T
    none_zero_col_index = np.mat(none_zero_position[1],dtype=int).T
    none_zero_position = np.hstack((none_zero_row_index,none_zero_col_index))
    none_zero_position = np.array(none_zero_position)
    name = 'miRNA_disease.txt'
    np.savetxt(name, none_zero_position,fmt="%d",delimiter=' ')

def get_embedding(vectors: dict):
    matrix = np.zeros((
        #len(vectors),
        726,
        len(list(vectors.values())[0])
    ))
    for key, value in vectors.items():
        matrix[int(key), :] = value
    return matrix

#获得miRNA_disease_emb
def Get_embedding_Matrix(miRNA_disease_matrix_net):
    Net2edgelist(miRNA_disease_matrix_net)

    graph1 = graph.Graph()
    graph1.read_edgelist("./miRNA_disease.txt")
    model = sdne.SDNE(graph1, [1000, 128])
    return get_embedding(model.vectors)

def model_evaluate(real_score,predict_score):

    AUPR = get_AUPR(real_score,predict_score)
    AUC = get_AUC(real_score,predict_score)
    [f1,accuracy,recall,spec,precision] = get_Metrics(real_score,predict_score)
    return np.array([AUPR,AUC,f1,accuracy,recall,spec,precision])

def get_AUPR(real_score, predict_score):
    sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholdlist = []
    for i in range(999):
        threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
        thresholdlist.append(threshold)
    thresholds = np.matrix(thresholdlist)
    TN = np.zeros((1, len(thresholdlist)))
    TP = np.zeros((1, len(thresholdlist)))
    FN = np.zeros((1, len(thresholdlist)))
    FP = np.zeros((1, len(thresholdlist)))
    for i in range(thresholds.shape[1]):
        p_index = np.where(predict_score >= thresholds[0, i])
        TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
        FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
        # print(TP[0, i], FP[0, i])
        n_index = np.where(predict_score < thresholds[0, i])
        FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
        TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    x = list(np.array(recall).flatten())
    y = list(np.array(precision).flatten())
    xy = [(x, y) for x, y in zip(x, y)]
    xy.sort()
    x = [x for x, y in xy]
    y = [y for x, y in xy]
    new_x = [x for x, y in xy]
    new_y = [y for x, y in xy]
    new_x[0] = 0
    new_y[0] = 1
    new_x.append(1)
    new_y.append(0)
    area = 0
    for i in range(thresholds.shape[1]):
        area = area + (new_y[i] + new_y[i + 1]) * (new_x[i + 1] - new_x[i]) / 2
    return area


def get_AUC(real_score, predict_score):
    sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholdlist = []
    for i in range(999):
        threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
        thresholdlist.append(threshold)
    thresholds = np.matrix(thresholdlist)
    TN = np.zeros((1, len(thresholdlist)))
    TP = np.zeros((1, len(thresholdlist)))
    FN = np.zeros((1, len(thresholdlist)))
    FP = np.zeros((1, len(thresholdlist)))
    for i in range(thresholds.shape[1]):
        p_index = np.where(predict_score >= thresholds[0, i])
        TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
        FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
        n_index = np.where(predict_score < thresholds[0, i])
        FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
        TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    x = list(np.array(1 - spe).flatten())
    y = list(np.array(sen).flatten())
    xy = [(x, y) for x, y in zip(x, y)]
    xy.sort()
    new_x = [x for x, y in xy]
    new_y = [y for x, y in xy]
    new_x[0] = 0
    new_y[0] = 0
    new_x.append(1)
    new_y.append(1)
    # print(list(np.array(new_x).flatten()))
    # print(list(np.array(new_y).flatten()))
    area = 0
    for i in range(thresholds.shape[1]):
        area = area + (new_y[i] + new_y[i + 1]) * (new_x[i + 1] - new_x[i]) / 2
    return area


def get_Metrics(real_score, predict_score):
    sorted_predict_score = sorted(list(set(np.array(predict_score).flatten())))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholdlist = []
    for i in range(999):
        threshold = sorted_predict_score[int(math.ceil(sorted_predict_score_num * (i + 1) / 1000) - 1)]
        thresholdlist.append(threshold)
    thresholds = np.matrix(thresholdlist)
    TN = np.zeros((1, len(thresholdlist)))
    TP = np.zeros((1, len(thresholdlist)))
    FN = np.zeros((1, len(thresholdlist)))
    FP = np.zeros((1, len(thresholdlist)))
    for i in range(thresholds.shape[1]):
        p_index = np.where(predict_score >= thresholds[0, i])
        TP[0, i] = len(np.where(real_score[p_index] == 1)[0])
        FP[0, i] = len(np.where(real_score[p_index] == 0)[0])
        n_index = np.where(predict_score < thresholds[0, i])
        FN[0, i] = len(np.where(real_score[n_index] == 1)[0])
        TN[0, i] = len(np.where(real_score[n_index] == 0)[0])
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sen = TP / (TP + FN)
    recall = sen
    spec = TN / (TN + FP)
    precision = TP / (TP + FP)
    f1 = 2 * recall * precision / (recall + precision)
    max_index = np.argmax(f1)
    max_f1 = f1[0, max_index]
    max_accuracy = accuracy[0, max_index]
    max_recall = recall[0, max_index]
    max_spec = spec[0, max_index]
    max_precision = precision[0, max_index]
    return [max_f1, max_accuracy, max_recall, max_spec, max_precision]

def constructNet(miRNA_dis_matrix):
    miRNA_matrix = np.matrix(np.zeros((miRNA_dis_matrix.shape[0], miRNA_dis_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(np.zeros((miRNA_dis_matrix.shape[1], miRNA_dis_matrix.shape[1]),dtype=np.int8))

    mat1 = np.hstack((miRNA_matrix,miRNA_dis_matrix))
    mat2 = np.hstack((miRNA_dis_matrix.T,dis_matrix))

    return np.vstack((mat1,mat2))

def cross_validation_experiment(miRNA_dis_matrix,miRNA_fam_matrix,dis_sim_matrix,seed):
    none_zero_position = np.where(miRNA_dis_matrix != 0)
    none_zero_row_index = none_zero_position[0]
    none_zero_col_index = none_zero_position[1]

    zero_position = np.where(miRNA_dis_matrix == 0)
    zero_row_index = zero_position[0]
    zero_col_index = zero_position[1]

    np.random.seed(seed)
    positive_randomlist = [i for i in range(len(none_zero_row_index))]
    random.shuffle(positive_randomlist)

    metric = np.zeros((1, 7))
    k_folds = 5
    print("seed=%d, evaluating miRNA-disease...." % (seed))

    for k in range(k_folds):
        print("------this is %dth cross validation------"%(k+1))
        if k != k_folds-1:
            positive_test = positive_randomlist[k*int(len(none_zero_row_index)/k_folds):(k+1)*int(len(none_zero_row_index)/k_folds)]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))
        else:
            positive_test = positive_randomlist[k * int(len(none_zero_row_index) / k_folds)::]
            positive_train = list(set(positive_randomlist).difference(set(positive_test)))

        positive_test_row = none_zero_row_index[positive_test]
        positive_test_col = none_zero_col_index[positive_test]

        positive_train_row = none_zero_row_index[positive_train]
        positive_train_col = none_zero_col_index[positive_train]
        # train_row = np.append(positive_train_row, zero_row_index)
        # train_col = np.append(positive_train_col, zero_col_index)

        train_row = np.append(positive_train_row, positive_test_row)
        train_col = np.append(positive_train_col, positive_test_col)
        train_row = np.append(train_row, zero_row_index)
        train_col = np.append(train_col, zero_col_index)

        train_miRNA_dis_matrix = np.copy(miRNA_dis_matrix)
        train_miRNA_dis_matrix[positive_test_row, positive_test_col] = 0

        miRNA_disease_matrix_net = constructNet(train_miRNA_dis_matrix)
        miRNA_disease_emb = Get_embedding_Matrix(np.mat(miRNA_disease_matrix_net))
        miRNA_len = miRNA_dis_matrix.shape[0]

        miRNA_emb_matrix = np.array(miRNA_disease_emb[0:miRNA_len, 0:])
        dis_emb_matrix = np.array(miRNA_disease_emb[miRNA_len::, 0:])

        train_feature_matrix = []
        train_label_vector = []

        for num in range(len(train_row)):
            feature_vector = np.append(np.append(miRNA_emb_matrix[train_row[num], :], dis_emb_matrix[train_col[num], :]),
                                                 np.append(miRNA_fam_matrix[train_row[num], :],dis_sim_matrix[train_col[num], :]))
            train_feature_matrix.append(feature_vector)
            train_label_vector.append(train_miRNA_dis_matrix[train_row[num], train_col[num]])

        test_feature_matrix = []
        test_label_vector = []

        test_position = np.where(train_miRNA_dis_matrix == 0)
        test_row = test_position[0]
        test_col = test_position[1]

        for num in range(len(test_row)):
            feature_vector = np.append(np.append(miRNA_emb_matrix[test_row[num], :], dis_emb_matrix[test_col[num], :]),
                                                 np.append(miRNA_fam_matrix[test_row[num], :],dis_sim_matrix[test_col[num], :]))
            test_feature_matrix.append(feature_vector)
            test_label_vector.append(miRNA_dis_matrix[test_row[num], test_col[num]])

        train_feature_matrix = np.array(train_feature_matrix)
        train_label_vector = np.array(train_label_vector)
        test_feature_matrix = np.array(test_feature_matrix)
        test_label_vector = np.array(test_label_vector)

        clf = RandomForestClassifier(random_state=1, n_estimators=350, oob_score=False, n_jobs=-1)
        clf.fit(train_feature_matrix, train_label_vector)
        predict_y_proba = clf.predict_proba(test_feature_matrix)[:, 1]
        predict_y_proba = np.array(predict_y_proba)

        metric += model_evaluate(test_label_vector, predict_y_proba)

        del train_feature_matrix
        del train_label_vector
        del test_feature_matrix
        del test_label_vector
        gc.collect()

    print(metric / k_folds)

    metric = np.array(metric / k_folds)

    name = 'result/miRNA_disease_seed=' + str(seed) + '.csv'
    np.savetxt(name, metric, delimiter=',')

    return metric

if __name__=="__main__":
    datetime1 = datetime.now()
    miRNA_dis_matrix = np.loadtxt('data/miRNA_disease.csv', delimiter=',', dtype=float)
    miRNA_fam_matrix = np.loadtxt('data/miRNA_family.csv',delimiter=',', dtype=float)
    dis_sim_matrix = np.loadtxt('data/disease_similarity.csv', delimiter=',', dtype=float)

    result = np.zeros((1, 7), float)
    average_result = np.zeros((1, 7), float)
    circle_time = 10

    for i in range(circle_time):
        result += cross_validation_experiment(miRNA_dis_matrix, miRNA_fam_matrix, dis_sim_matrix,i)

    average_result = result / circle_time
    print(average_result)
    np.savetxt('result/avg_miRNA_disease.csv', average_result, delimiter=',')
    print(datetime.now() - datetime1)
