import argparse
import random
import matplotlib.pyplot as plt
import numpy as np

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", dest="path", type=str, help="Path of Input File")
    parser.add_argument("-b", dest="base", type=int, default=3, help="Polynomial base")
    parser.add_argument("-l", dest="lamb", type=int, default=0, help="Lambda of LSE")
    parser.add_argument("-i", dest="itr", type=int, default=5, help="Iteration of Newton's Method")
    args = parser.parse_args()

    return args

def read_data(path):    # Read the Input File
    input_data = open(path, 'r')
    data_points = list()
    for data_point in input_data:
        data_array = list(data_point.strip().split(','))
        data_points.append(data_array)

    return data_points

def lse(data_points, base, lamb):   # Calculate the Less Square Error
    A, Input, b = Arg_Matrix(data_points, base)
    I = Identity(base)

    A = np.array(A)
    A_reverse = A.T
    result = np.dot(A_reverse, A) + np.multiply(I, lamb)
    result = Ieverse(result, I, base)

    result = np.dot(result, A_reverse)
    result = np.dot(result, b)

    result_poly = Polynomial(result, base)
    result_error = Total_error(result, base, Input, b)

    return result, result_poly, result_error

def Newton(data_points, base, iterations):
    A, Input, b = Arg_Matrix(data_points, base)
    I = Identity(base)
    x = []

    for index in range(base):
        x.append(random.randint(1, 100))

    A = np.array(A)
    A_reverse = A.T

    for iteration in range(iterations):
        hessiian = np.dot(A_reverse, A)
        gradient_l = np.dot(hessiian, x)
        gradient_l = np.multiply(gradient_l, 2)

        gradient_r = np.dot(A_reverse, b)
        gradient_r = np.multiply(gradient_r, 2)

        gradient = gradient_l - gradient_r

        hessiian = np.multiply(hessiian, 2)
        hessiian_inverse = Ieverse(hessiian, I, base)

        x = x - np.dot(hessiian_inverse, gradient)

    result = x

    result_poly = Polynomial(result, base)
    result_error = Total_error(result, base, Input, b)

    return result, result_poly, result_error


def Arg_Matrix(data_points, base):  # Create Arguement Matrix
    A = []
    Input = []
    b = []
    for data_point in data_points:
        A_row = []
        for power in range(base-1, -1, -1):
            A_row.append(float(data_point[0]) ** power)
        A.append(A_row)
        Input.append(float(data_point[0]))
        b.append(float(data_point[1]))

    return A, Input, b

def Identity(dim):  # Create Identity Matrix
    I = []
    for row in range(dim):
        I_row = []
        for col in range(dim):
            if row == col:
                I_row.append(1)
            else:
                I_row.append(0)
        I.append(I_row)

    return I

def Ieverse(matrix, I, dim):
    X = np.zeros((dim, dim), dtype=float)
    Y = np.zeros((dim, dim), dtype=float)
    L, U = LUDecompose(matrix, dim)

    for col in range(dim):
        for row in range(dim):
            Y[row][col]= I[row][col]
            for index in range(row):
                Y[row][col] = Y[row][col] - (Y[index][col] * L[row][index])

    for col in range(dim):
        for row in range(dim-1, -1, -1):
            X[row][col] = Y[row][col]
            for index in range(dim-1, row, -1):
                X[row][col] = X[row][col] - (X[index][col] * U[row][index])
            X[row][col]= X[row][col] / U[row][row]

    return X

def LUDecompose(matrix, dim):
    L = np.zeros((dim, dim))
    U = np.zeros((dim, dim))

    for i in range(dim):
        for k in range(i, dim):
            sum = 0
            for j in range(i):
                sum += (L[i][j] * U[j][k])
            U[i][k] = matrix[i][k] - sum

        for k in range(i, dim):
            if (i == k):
                L[i][i] = 1
            else:
                sum = 0
                for j in range(i):
                    sum += (L[k][j] * U[j][i])
                L[k][i] = (matrix[k][i] - sum) /U[i][i]
    return L, U

def plot_result(data_points, base, result):   # print the Result
    data_points_x = []
    data_points_y = []

    for data_point in data_points:
        data_points_x.append(float(data_point[0]))
        data_points_y.append(float(data_point[1]))

    sample_x = np.linspace(min(data_points_x)-1, max(data_points_x)+1, 50)

    for sub_plot in range(2):
        sample_y = []
        for sample_point in sample_x:
            sample_error = float(0)
            for power in range(base - 1, -1, -1):
                sample_error = sample_error + result[sub_plot][base - power - 1] * (sample_point ** power)
            sample_y.append(sample_error)

        plt.subplot(2, 1, sub_plot+1)
        plt.plot(data_points_x, data_points_y, 'ro')
        plt.plot(sample_x, sample_y, '-k')
        plt.xlim(min(data_points_x)-1, max(data_points_x)+1)

    plt.show()

def show_result(way, poly, error): # show the Result's Value
    print(f'{way}:')
    print(f'Fitting line: {poly}')
    print(f'Total error: {error}')

def Polynomial(result, base):
    poly = ''
    for power in range(base-1, -1, -1):
        if power != 0:
            poly = poly + str(result[base - power - 1]) + 'X^' + str(power) + ' + '
        else:
            poly = poly + str(result[base - power - 1])

    return poly

def Total_error(result, base, input, label):
    error = float(0)
    for sample in range(len(label)):
        sample_error = float(0)
        for power in range(base-1, -1, -1):
            sample_error = sample_error + result[base - power - 1] * (input[sample] ** power)
        sample_error = (sample_error - label[sample]) ** 2
        error = error + sample_error

    return error

if __name__ == "__main__":
    args = set_args()
    data_points = read_data(args.path)

    lse_args, lse_poly, lse_error = lse(data_points, args.base, args.lamb)
    show_result('LSE', lse_poly, lse_error)

    print()

    newton_args, newton_poly, newton_error = Newton(data_points, args.base, args.itr)
    show_result("Newton's Method", newton_poly, newton_error)

    plot_result(data_points, args.base, np.array([lse_args,newton_args]))

