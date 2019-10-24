import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt

filepath = 'E:/Workspace/GANproject/'
x_data = np.load(filepath + 'x_edata.npy')
y_data = np.load(filepath + 'y_edata.npy')


# 生成数据集
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.3, random_state=0)

#设置参数
params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'gamma': 0.1,
    'max_depth': 20,
    'lambda': 1,
    'subsample': 0.5,
    'colsample_bytree': 1,
    'min_child_weight': 1,
    'verbosity': 2,
    'eta': 0.1,
    'seed': 0,
    'eval_metric': ['rmse', 'error', 'logloss']
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test, y_test)
num_round = 100
watchlist = [(dtest, 'eval'), (dtrain, 'train')]
result = {}

#训练模型
bst = xgb.train(params, dtrain, num_round, watchlist,
                evals_result=result, early_stopping_rounds=10)
bst.save_model(filepath + 'xgb/xgboost.lastRound.model')

plt.figure(figsize=(10, 15))
plt.subplot(311)
plt.plot(result['eval']['rmse'], label='Eval', color='r')
plt.plot(result['train']['rmse'], label='Train', color='c')
plt.xlabel('Round')
plt.ylabel('RMSE')
plt.grid()
plt.legend()

plt.subplot(312)
plt.plot(np.array(result['eval']['logloss']), label='Eval', color='r')
plt.plot(np.array(result['train']['logloss']), label='Train', color='c')
plt.xlabel('Round')
plt.ylabel('Logloss')
plt.grid()
plt.legend()

plt.subplot(313)
plt.plot(np.array(result['eval']['error']), label='Eval', color='r')
plt.plot(np.array(result['train']['error']), label='Train', color='c')
plt.xlabel('Round')
plt.ylabel('Error')
plt.grid()
plt.legend()

plt.savefig(filepath + 'xgb/xgbCurve.svg')
plt.show()
