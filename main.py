import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cvxpy as cp

def Load_Dataset(path):
    #df = pd.read_csv(path, header=None)
    df = pd.read_csv(path, usecols=[0,1,2], names=['colA', 'colB','colC'], header=None)
    X2 = df[['colA', 'colB']]
    Y2 = df['colC']
    return X2.to_numpy(), Y2.to_numpy()

def SVM_Problem(X, Y, beta0, beta, e,C):
    objective = cp.Minimize(0.5 * (cp.sum_squares(beta)) + (C * cp.sum(e)))
    constraints = [0 <= e, cp.multiply(Y, X @ beta + beta0) >= 1 - e]
    prob = cp.Problem(objective, constraints)
    result = prob.solve(solver=cp.ECOS)
    return result

def SVM_Error():
    X, Y = Load_Dataset("xy_test.csv")
    p = 2
    n = 1000
    beta0 = cp.Variable()
    beta = cp.Variable(p)
    e = cp.Variable(n)
    a = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    length = len(a)
    Cx = []
    problem_solved_solution = []
    for i in range(length):
        x = 2 ** a[i]
        Cx.append(x)
    print(Cx)
    error_list = []
    for i in range(length):
        r = SVM_Problem(X, Y, beta0, beta, e, Cx[i])
        problem_solved_solution.append(r)
        y_pred = X @ beta.value
        y_pred = y_pred.astype(int)
        error_list.append(np.sum(np.square((Y - y_pred)*(1/2*n))))
    return error_list





def main():
    #i
    X, Y = Load_Dataset("xy_train.csv")
    print (X)
    print (Y)
    p = 2
    n = 200
    beta0 = cp.Variable()
    beta = cp.Variable(p)
    e = cp.Variable(n)
    C=1
    problem_result = SVM_Problem(X, Y, beta0, beta, e,C)
    a0 = beta0.value
    a1 = beta.value
    a2 = e.value
    print("The result is: ",problem_result)
    print("The Amount of beta0:", a0)
    print("The Amount of beta:", a1)
    print("The Amount of Epsilon:", a2)

    #ii
    b1 = beta.value[0]
    b2 = beta.value[1]

    A = X[:, 0]
    print(A)
    B = X[:, 1]
    print(B)

    fig ,ax = plt.subplots()
    ax.scatter(A, B, c=Y, cmap=plt.cm.RdGy)
    plt.xlim(-4,6)
    plt.ylim(-4,6)
    ax.set_title("Desision Boundry for Features")
    plt.plot(A, (-b1 * A - a0) / b2, color='b')
    plt.show()

    #iv
    C = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32]
    error_list=SVM_Error()

    plt.plot(C, error_list)
    plt.title("Error Plot")
    plt.xlabel("C")
    plt.ylabel("Miscllasification Error")
    plt.show()





if __name__ == '__main__':
    main()


"""

b0=beta0.value



# Generate data for SVM classifier with L1 regularization.
import numpy as np
import pandas as pd
import cvxpy as cp

# creating data


df = pd.read_csv('xy_train.csv', header=None)
print(df)
data = np.array(df)
onez = np.ones((200, 1))
X = np.concatenate((onez, data[:, 0:2]), axis=1)
print(X)
Y = data[:, 2:3]
# X = Y = data[:, 0:2]

p = 3
n = 200

beta = cp.Variable((p, 1))
ep = cp.Variable((1, n))
beta_z = cp.Variable()
# l = cp.multiply(Y, X @ beta + beta_z)
c = cp.Variable()
c = 1
objective = cp.Minimize(1 / 2 * (cp.sum_squares(beta)) + (c * cp.sum(ep)))
constraints = [0 <= ep, cp.multiply(Y, X @ beta + beta_z) >= 1 - ep]
prob = cp.Problem(objective, constraints)
result = prob.solve()
print("ep:", ep.value[0][0])
print("\n beta:", beta.value)
print("\n beta_0:", beta_z.value)





from __future__ import division




import cvxpy as cp
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

def load_dataset(path):
    df = pd.read_csv(path)
    xs = df.iloc[:,:2]
    ys = df.iloc[:,2]
    return xs, ys

x,y = load_dataset("xy_train.csv")
print (x , y)
n = 3
m = 200
# Form SVM with L1 regularization problem.
beta = cp.Variable((n,1))
ep = cp.Variable((m,1))
beta0= cp.Variable()
c=1
loss =cp.Minimize(1/2 * (cp.sum_squares(beta)) + (c*cp.sum(ep)))
constraints = [ep >= 0,
               cp.multiply(y, (((x.T) @ beta) + beta0)) >= 1 - ep]

prob = cp.Problem(loss , constraints)
probe.solve()
print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var", x.value, y.value)
print("\n beta:",beta.value)
print("\n beta_0:",beta0.value)

TRIALS = 100
train_error = np.zeros(TRIALS)
test_error = np.zeros(TRIALS)
lambda_vals = np.logspace(-2, 0, TRIALS)
beta_vals = []
for i in range(TRIALS):
    lambd.value = lambda_vals[i]
    prob.solve()
    train_error[i] = (np.sign(X.dot(beta_true) + offset) != np.sign(X.dot(beta.value) - v.value)).sum()/m
    test_error[i] = (np.sign(X_test.dot(beta_true) + offset) != np.sign(X_test.dot(beta.value) - v.value)).sum()/TEST
    beta_vals.append(beta.value)

# Plot the train and test error over the trade-off curve.


plt.plot(lambda_vals, train_error, label="Train error")
plt.plot(lambda_vals, test_error, label="Test error")
plt.xscale('log')
plt.legend(loc='upper left')
plt.xlabel(r"$\lambda$", fontsize=16)
plt.show()

# Plot the regularization path for beta.
for i in range(n):
    plt.plot(lambda_vals, [wi[i,0] for wi in beta_vals])
plt.xlabel(r"$\lambda$", fontsize=16)
plt.xscale("log")
"""