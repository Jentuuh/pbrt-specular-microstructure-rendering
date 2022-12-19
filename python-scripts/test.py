from threading import Thread
import numpy as np
import random


def thread_function(threadNumber, matrix):
    test_matrix = np.identity(4)
    np.linalg.inv(test_matrix)
    np.linalg.inv(test_matrix)
    np.linalg.inv(test_matrix)
    np.linalg.inv(test_matrix)

    for i in range(100):
        matrix[threadNumber, i] += random.randint(0, 10)



result = np.zeros((100, 100))
threads = [None] * 100

for i in range(100):
        threads[i] = Thread(target=thread_function, args=(i, result))
        threads[i].start()

for i in range(100):
    threads[i].join()
print(result)
