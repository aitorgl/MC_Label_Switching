import os
from collections import Counter
from collections import defaultdict
from copy import deepcopy

import numpy as np

from sklearn.utils import check_random_state

class ECOC():
    """
    ECOC (Error Correcting Output Codes) is ensemble method for multi-class classification problems.
    Each class is encoded with unique binary or ternary code (where 0 means that class is excluded from training set
    of binary classifier). Then in the learning phase each binary classifier is learned. In the decoding phase the class
    which is closest to test instance in the sense of Hamming distance is chosen.
    """

    _allowed_encodings = ['dense', 'sparse', 'complete', 'OVA', 'OVO']
    _allowed_weights = [None, 'acc', 'avg_tpr_min']

    def __init__(self,  encoding='OVO', labels=None, weights=None):
        """
        :param encoding:
            algorithm for encoding classes. Possible encodings:

            * 'dense':
                ceil(10log2(num_of_classes)) dichotomies, -1 and 1 with probability 0.5 each
            * 'sparse' :
                ceil(10log2(num_of_classes)) dichotomies, 0 with probability 0.5, -1 and 1 with probability 0.25 each
            * 'OVO' :
                'one vs one' - n(n-1)/2 dichotomies, where n is number of classes, one for each pair of classes.
                Each column has one 1 and one -1 for classes included in particular pair, 0s for remaining classes.
            * 'OVA' :
                'one vs all' - number of dichotomies is equal to number of classes. Each column has one 1 and
                -1 for all remaining rows
            * 'complete' : 2^(n-1)-1 dichotomies, reference
                T. G. Dietterich and G. Bakiri.
                Solving multiclass learning problems via error-correcting output codes.
                Journal of Artificial Intelligence Research, 2:263–286, 1995.
        :param n_neighbors:
        :param weights:
            strategy for dichotomies weighting. Possible values:

            * None :
                no weighting applied
            * 'acc' :
                accuracy-based weights
            * 'avg_tpr_min' :
                weights based on average true positive rates of dichotomies
        """
        super().__init__()
        self.encoding = encoding
        self.weights = weights


        self._code_matrix = None
        self._labels = labels
        self._dich_weights = None
        self._gen_code_matrix()

    def _gen_code_matrix(self):
        if self.encoding == 'dense':
            self._code_matrix = self._encode_dense(self._labels.shape[0])
        elif self.encoding == 'sparse':
            self._code_matrix = self._encode_sparse(self._labels.shape[0])
        elif self.encoding == 'complete':
            self._code_matrix = self._encode_complete(self._labels.shape[0])
        elif self.encoding == 'OVO':
            self._code_matrix = self._encode_ovo(self._labels.shape[0])
        elif self.encoding == 'OVA':
            self._code_matrix = self._encode_ova(self._labels.shape[0])
        else:
            raise ValueError("Unknown matrix generation encoding: %s, expected to be one of %s."
                             % (self.encoding, ECOC._allowed_encodings))

    def _encode_dense(self, number_of_classes, random_state=0, number_of_code_generations=10000):
        try:
            dirname = os.path.dirname(__file__)
            matrix = np.load(dirname + f'/cached_matrices/dense_{number_of_classes}.npy')
            return matrix
        except IOError:
            print(f'Could not find cached matrix for dense code for {number_of_classes} classes, generating matrix...')

        number_of_columns = int(np.ceil(10 * np.log2(number_of_classes)))
        code_matrix = np.ones((number_of_classes, number_of_columns))
        random_state = check_random_state(random_state)

        max_min_dist = 0
        for i in range(number_of_code_generations):
            tmp_code_matrix = np.ones((number_of_classes, number_of_columns))
            min_dist = float('inf')

            for row in range(0, number_of_classes):
                for col in range(0, number_of_columns):
                    if random_state.randint(0, 2) == 1:
                        tmp_code_matrix[row, col] = -1

                for compared_row in range(0, row):
                    dist = self._hamming_distance(tmp_code_matrix[compared_row], tmp_code_matrix[row])
                    if dist < min_dist:
                        min_dist = dist

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                code_matrix = tmp_code_matrix
        return code_matrix

    def _encode_sparse(self, number_of_classes, random_state=0, number_of_code_generations=10000):
        try:
            dirname = os.path.dirname(__file__)
            matrix = np.load(dirname + f'/cached_matrices/sparse_{number_of_classes}.npy')
            return matrix
        except IOError:
            print(f'Could not find cached matrix for sparse code for {number_of_classes} classes, generating matrix...')

        number_of_columns = int(np.ceil(15 * np.log2(number_of_classes)))
        code_matrix = np.ones((number_of_classes, number_of_columns))
        random_state = check_random_state(random_state)

        max_min_dist = 0
        for i in range(number_of_code_generations):
            tmp_code_matrix = np.ones((number_of_classes, number_of_columns))
            min_dist = float('inf')

            for row in range(0, number_of_classes):
                for col in range(0, number_of_columns):
                    rand = random_state.randint(0, 4)
                    if rand < 2:
                        tmp_code_matrix[row, col] = 0
                    elif rand == 3:
                        tmp_code_matrix[row, col] = -1

                if np.count_nonzero(tmp_code_matrix[row]) == 0:
                    break

                for compared_row in range(0, row):
                    dist = self._hamming_distance(tmp_code_matrix[compared_row], tmp_code_matrix[row])
                    if dist < min_dist:
                        min_dist = dist

            if self._has_matrix_all_zeros_column(tmp_code_matrix):
                continue

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                code_matrix = tmp_code_matrix

        return code_matrix

    def _encode_ova(self, number_of_classes):
        matrix = np.identity(number_of_classes)
        matrix[matrix == 0] = -1
        return matrix

    def _encode_ovo(self, number_of_classes):
        number_of_columns = int(number_of_classes * (number_of_classes - 1) / 2)
        matrix = np.zeros((number_of_classes, number_of_columns), dtype=int)
        indices_map = self._map_indices_to_class_pairs(number_of_classes)
        for row in range(number_of_classes):
            for col in range(number_of_columns):
                if indices_map[col][0] == row:
                    matrix[row, col] = 1
                elif indices_map[col][1] == row:
                    matrix[row, col] = -1
        return matrix

    def _map_indices_to_class_pairs(self, number_of_classes):
        indices_map = dict()
        idx = 0
        for i in range(number_of_classes):
            for j in range(i + 1, number_of_classes):
                indices_map[idx] = (i, j)
                idx += 1
        return indices_map

    def _encode_complete(self, number_of_classes):
        code_length = 2 ** (number_of_classes - 1) - 1
        matrix = np.ones((number_of_classes, code_length))
        for row_idx in range(1, number_of_classes):
            digit = -1
            partial_code_len = 2 ** (number_of_classes - row_idx - 1)
            for idx in range(0, code_length, partial_code_len):
                matrix[row_idx][idx:idx + partial_code_len] = digit
                digit *= -1
        return matrix

    def _hamming_distance(self, v1, v2):
        return np.count_nonzero(v1 != v2)

    def _has_matrix_all_zeros_column(self, matrix):
        return (~matrix.any(axis=0)).any()

    def _get_closest_class(self, row):
        if self.weights is not None:
            return self._labels[
                np.argmin(
                    [sum(np.multiply(self.dich_weights, (encoded_class - row) ** 2)) for encoded_class in
                     self._code_matrix])]
        else:
            return self._labels[
                np.argmin([self._hamming_distance(row, encoded_class) for encoded_class in self._code_matrix])]


    def _calc_weights(self, X_for_weights, y_for_weights):
        if self.weights not in ECOC._allowed_weights:
            raise ValueError("Unknown weighting strategy: %s, expected to be one of %s."
                             % (self.weights, ECOC._allowed_weights))

        dich_weights = np.ones(self._code_matrix.shape[1])
        if self.weights == 'acc':
            for clf_idx, clf in enumerate(self._binary_classifiers):
                samples_no = 0
                correct_no = 0
                for sample, sample_label in zip(X_for_weights, y_for_weights):
                    if self._code_matrix[np.where(self._labels == sample_label)[0][0]][clf_idx] != 0:
                        samples_no += 1
                        if clf.predict([sample])[0] == \
                                self._code_matrix[np.where(self._labels == sample_label)[0][0]][clf_idx]:
                            correct_no += 1
                if samples_no != 0:
                    acc = correct_no / samples_no
                    dich_weights[clf_idx] = -1 + 2 * acc
        elif self.weights == 'avg_tpr_min':
            min_counter = Counter([y for y in y_for_weights if y in self.minority_classes])

            for clf_idx, clf in enumerate(self._binary_classifiers):
                min_correct_pred = defaultdict(lambda: 0)
                for sample, sample_label in zip(X_for_weights, y_for_weights):
                    if clf.predict([sample])[0] == \
                            self._code_matrix[np.where(self._labels == sample_label)[0][0]][clf_idx]:
                        min_correct_pred[sample_label] += 1
                avg_tpr_min = np.mean([min_correct_pred[clazz] / min_counter[clazz] for clazz in min_counter.keys()])
                dich_weights[clf_idx] = avg_tpr_min

        self.dich_weights = dich_weights
