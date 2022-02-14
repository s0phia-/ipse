import numpy as np
import stew.example_data as create
import stew.mlogit as mlogit
import stew.utils as utils
import matplotlib.pyplot as plt

#
# """
# With undirected data (that is, feature weights)
# """
# np.random.seed(1)
#
# num_features = 8
# num_choices = 20
# max_cv_splits = 10
# num_lambdas = 100
# num_states = 1000
# directed_beta = False
#
# data = create.discrete_choice_example_data(num_states=num_states*2, num_features=num_features, num_choices=num_choices,
#                                            probabilistic=True, directed_beta=directed_beta)
# data.shape
# trainingDataAll = data[data[:, 0] < num_states, :]
# trainingDataAll.shape
#
# testingData = data[data[:, 0] >= num_states, :]
# testingData.shape
#
# trueCrit = testingData[:, 1]
# testingData = np.delete(testingData, 1, 1)
# testingData.shape
#
# trainingDataSizes = np.array([5, 10, 20, 50, 100, 200, 400])
# len_trainingDataSizes = len(trainingDataSizes)
# unregularized_errors = np.ones(len_trainingDataSizes)
# cv_reg_errors = np.ones(len_trainingDataSizes)
#
# for trainingDataSizeIt in range(len_trainingDataSizes):  # trainingDataSizeIt = 0
#     trainingDataSize = trainingDataSizes[trainingDataSizeIt]
#     print("trainingDataSize", trainingDataSize)
#
#     trainingData = trainingDataAll.copy()[trainingDataAll[:, 0] < trainingDataSize, :]
#
#     # FIT with lambda = 0
#     mlog = mlogit.StewMultinomialLogit(num_features=num_features, max_splits=max_cv_splits, num_lambdas=num_lambdas, nonnegative=False)
#     weights = mlog.fit(data=trainingData, lam=0, standardize=False)
#     predicted_choices = mlog.predict(new_data=testingData, weights=weights)
#     unregularized_errors[trainingDataSizeIt] = utils.multi_class_error(predicted_choices, trueCrit)
#
#     # CV FIT
#     mlog = mlogit.StewMultinomialLogit(num_features=num_features, max_splits=max_cv_splits, num_lambdas=num_lambdas, nonnegative=False)
#     weights, cv_min_lambda = mlog.cv_fit(data=trainingData)
#     predicted_choices = mlog.predict(new_data=testingData, weights=weights)
#     cv_reg_errors[trainingDataSizeIt] = utils.multi_class_error(predicted_choices, trueCrit)
#
#
#
# fig1, ax1 = plt.subplots()
#
# ax1.plot(trainingDataSizes, unregularized_errors, label="unregularized")
# ax1.plot(trainingDataSizes, cv_reg_errors, label="cv_reg")
#
# plt.title('Test of cvfit')
# plt.xlabel('Train set size')
# plt.ylabel('Mean score')
# plt.legend()
# plt.show()


"""
Same with directed beta

"""

np.random.seed(1)

num_features = 8
num_choices = 20
max_cv_splits = 10
num_lambdas = 100
num_states = 1000
directed_beta = True

data = create.discrete_choice_example_data(num_states=num_states*2, num_features=num_features, num_choices=num_choices,
                                           probabilistic=True, directed_beta=directed_beta)
data.shape
trainingDataAll = data[data[:, 0] < num_states, :]
trainingDataAll.shape

testingData = data[data[:, 0] >= num_states, :]
testingData.shape

trueCrit = testingData[:, 1]
testingData = np.delete(testingData, 1, 1)
testingData.shape

trainingDataSizes = np.array([3, 5, 7, 10, 15, 20, 50, 100, 200])
len_trainingDataSizes = len(trainingDataSizes)
first_argmin_errors = np.ones(len_trainingDataSizes)
last_argmin_errors = np.ones(len_trainingDataSizes)


mlog_first_argmin = mlogit.StewMultinomialLogit(num_features=num_features, max_splits=max_cv_splits,
                                   num_lambdas=num_lambdas, nonnegative=False, lambda_max=6.0,
                                   verbose=True, last_argmin=False)

mlog_last_argmin = mlogit.StewMultinomialLogit(num_features=num_features, max_splits=max_cv_splits,
                                   num_lambdas=num_lambdas, nonnegative=False, lambda_max=6.0,
                                   verbose=True, last_argmin=True)


for trainingDataSizeIt in range(len_trainingDataSizes):  # trainingDataSizeIt = 0
    trainingDataSize = trainingDataSizes[trainingDataSizeIt]
    print("trainingDataSize", trainingDataSize)

    trainingData = trainingDataAll.copy()[trainingDataAll[:, 0] < trainingDataSize, :]

    # FIT with lambda = 0
    # mlog = mlogit.StewMultinomialLogit(num_features=num_features, max_splits=max_cv_splits, num_lambdas=num_lambdas, nonnegative=False)
    # weights = mlog.fit(data=trainingData, lam=0, standardize=False)
    # predicted_choices = mlog.predict(new_data=testingData, weights=weights)
    # unregularized_errors[trainingDataSizeIt] = utils.multi_class_error(predicted_choices, trueCrit)

    # CV FIT
    # mlog = mlogit.StewMultinomialLogit(num_features=num_features, max_splits=max_cv_splits, num_lambdas=num_lambdas, nonnegative=False)
    weights, cv_min_lambda = mlog_first_argmin.cv_fit(data=trainingData)
    predicted_choices = mlog_first_argmin.predict(new_data=testingData, weights=weights)
    first_argmin_errors[trainingDataSizeIt] = utils.multi_class_error(predicted_choices, trueCrit)

    weights, cv_min_lambda = mlog_last_argmin.cv_fit(data=trainingData)
    predicted_choices = mlog_last_argmin.predict(new_data=testingData, weights=weights)
    last_argmin_errors[trainingDataSizeIt] = utils.multi_class_error(predicted_choices, trueCrit)



fig1, ax1 = plt.subplots()

ax1.plot(trainingDataSizes, first_argmin_errors, label="first_argmin")
ax1.plot(trainingDataSizes, last_argmin_errors, label="last_argmin")

plt.title('Test of cvfit -- directed predictors.')
plt.xlabel('Train set size')
plt.ylabel('Mean score')
plt.legend()
plt.show()

# fig1.savefig(os.path.join(plots_path, "mean_performance"))
# plt.close()
# print(mlog.fit(data=data, lam=0, standardize=False))
# print(mlog.fit(data=data, lam=0, standardize=True))
#
# print(mlog.fit(data=data, lam=100, standardize=False))
# print(mlog.fit(data=data, lam=100, standardize=True))
# predicted_choices = mlog.predict(new_data=np.delete(data, 1, 1), weights=weights)
# utils.multi_class_error(predicted_choices, data[:, 1])
# # print(mlog.fit(data=data, start_weights=mlog.weights, lam=100000))
# weights, cv_min_lambda = mlog.cv_fit(data=data)




