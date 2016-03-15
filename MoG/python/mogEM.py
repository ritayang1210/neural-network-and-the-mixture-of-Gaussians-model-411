from kmeans import *
import sys
import matplotlib.pyplot as plt
from nn import *
plt.ion()

def mogEM(x, K, iters, minVary=0):
  """
  Fits a Mixture of K Gaussians on x.
  Inputs:
    x: data with one data vector in each column.
    K: Number of Gaussians.
    iters: Number of EM iterations.
    minVary: minimum variance of each Gaussian.

  Returns:
    p : probabilities of clusters.
    mu = mean of the clusters, one in each column.
    vary = variances for the cth cluster, one in each column.
    logProbX = log-probability of data after every iteration.
  """
  N, T = x.shape 
  ''
  # Initialize the parameters
  randConst = 1
  p = randConst + np.random.rand(K, 1)
  p = p / np.sum(p)
  mn = np.mean(x, axis=1).reshape(-1, 1)
  vr = np.var(x, axis=1).reshape(-1, 1)
 
  # Change the initializaiton with Kmeans here
  #--------------------  Add your code here --------------------  
  # mu = mn + np.random.randn(N, K) * (np.sqrt(vr) / randConst)
  
  mu = KMeans(x, K, iters)
  
  #------------------------------------------------------------  
  vary = vr * np.ones((1, K)) * 2
  vary = (vary >= minVary) * vary + (vary < minVary) * minVary

  logProbX = np.zeros((iters, 1))

  # Do iters iterations of EM
  for i in xrange(iters):
    # Do the E step
    respTot = np.zeros((K, 1))
    respX = np.zeros((N, K))
    respDist = np.zeros((N, K))
    logProb = np.zeros((1, T))
    ivary = 1 / vary
    logNorm = np.log(p) - 0.5 * N * np.log(2 * np.pi) - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1)
    logPcAndx = np.zeros((K, T))
    for k in xrange(K):
      dis = (x - mu[:,k].reshape(-1, 1))**2
      logPcAndx[k, :] = logNorm[k] - 0.5 * np.sum(ivary[:,k].reshape(-1, 1) * dis, axis=0)
    
    mxi = np.argmax(logPcAndx, axis=1).reshape(1, -1) 
    mx = np.max(logPcAndx, axis=0).reshape(1, -1)
    PcAndx = np.exp(logPcAndx - mx)
    Px = np.sum(PcAndx, axis=0).reshape(1, -1)
    PcGivenx = PcAndx / Px
    logProb = np.log(Px) + mx
    logProbX[i] = np.sum(logProb)

    print 'Iter %d logProb %.5f' % (i, logProbX[i])

    # Plot log prob of data
    plt.figure(1);
    plt.clf()
    plt.plot(np.arange(i), logProbX[:i], 'r-')
    plt.title('Log-probability of data versus 30 iterations of EM(original initialization)')
    plt.xlabel('Iterations of EM')
    plt.ylabel('log P(D)');
    plt.draw()

    respTot = np.mean(PcGivenx, axis=1).reshape(-1, 1)
    respX = np.zeros((N, K))
    respDist = np.zeros((N,K))
    for k in xrange(K):
      respX[:, k] = np.mean(x * PcGivenx[k,:].reshape(1, -1), axis=1)
      respDist[:, k] = np.mean((x - mu[:,k].reshape(-1, 1))**2 * PcGivenx[k,:].reshape(1, -1), axis=1)

    # Do the M step
    p = respTot
    mu = respX / respTot.T
    vary = respDist / respTot.T
    vary = (vary >= minVary) * vary + (vary < minVary) * minVary
  
  return p, mu, vary, logProbX

def mogLogProb(p, mu, vary, x):
  """Computes logprob of each data vector in x under the MoG model specified by p, mu and vary."""
  K = p.shape[0]
  N, T = x.shape
  ivary = 1 / vary
  logProb = np.zeros(T)
  for t in xrange(T):
    # Compute log P(c)p(x|c) and then log p(x)
    logPcAndx = np.log(p) - 0.5 * N * np.log(2 * np.pi) \
        - 0.5 * np.sum(np.log(vary), axis=0).reshape(-1, 1) \
        - 0.5 * np.sum(ivary * (x[:, t].reshape(-1, 1) - mu)**2, axis=0).reshape(-1, 1)

    mx = np.max(logPcAndx, axis=0)
    logProb[t] = np.log(np.sum(np.exp(logPcAndx - mx))) + mx;
  return logProb

def q2():
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)
  iters = 20
  minVary = 0.01
  K = 2
  p2, mu2, vary2, logProbX2 = mogEM(train2, K, iters, minVary)
  ShowMeans(mu2)
  ShowMeans(vary2)
  print logProbX2[-1]

  p3, mu3, vary3, logProbX3 = mogEM(train3, K, iters, minVary)
  ShowMeans(mu3)
  ShowMeans(vary3)
  print logProbX3[-1]



def q3():
  iters = 30
  minVary = 0.01
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  # Train a MoG model with 20 components on all 600 training
  # vectors, with both original initialization and kmeans initialization.
  #------------------- Add your code here ---------------------
  K = 20
  p, mu, vary, logProbX = mogEM(inputs_train, K, iters, minVary)

  raw_input('Press Enter to continue.')


def q4():
  iters = 10
  minVary = 0.01
  errorTrain = np.zeros(4)
  errorTest = np.zeros(4)
  errorValidation = np.zeros(4)
  print(errorTrain)
  numComponents = np.array([2, 5, 15, 25])
  T = numComponents.shape[0]  
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)

  for t in xrange(T): 
    K = numComponents[t]
    # Train a MoG model with K components for digit 2
    #-------------------- Add your code here --------------------------------
    p2, mu2, vary2, logProbX2 = mogEM(train2, K, iters, minVary)
    
    # Train a MoG modewithl  K components for digit 3
    #-------------------- Add your code here --------------------------------
    p3, mu3, vary3, logProbX3 = mogEM(train3, K, iters, minVary)

    # Caculate the probability P(d=1|x) and P(d=2|x),
    # classify examples, and compute error rate
    # Hints: you may want to use mogLogProb function
    #-------------------- Add your code here --------------------------------
    logProb2_train = mogLogProb(p2, mu2, vary2, inputs_train)
    logProb3_train = mogLogProb(p3, mu3, vary3, inputs_train)
    errorTrain[t] = computeErrorRate(logProb2_train, logProb3_train, target_train)

    logProb2_valid = mogLogProb(p2, mu2, vary2, inputs_valid)
    logProb3_valid = mogLogProb(p3, mu3, vary3, inputs_valid)
    errorValidation[t] = computeErrorRate(logProb2_valid, logProb3_valid, target_valid)

    logProb2_test = mogLogProb(p2, mu2, vary2, inputs_test)
    logProb3_test = mogLogProb(p3, mu3, vary3, inputs_test)
    errorTest[t] = computeErrorRate(logProb2_test, logProb3_test, target_test)
    # count3 = np.count_nonzero(target_train)
    # countTotal = target_train.shape[1]
    # count2 = countTotal - count3
    # prob2 = count2/countTotal
    # prob3 = count3/countTotal

  # Plot the error rate
  plt.clf()
  #-------------------- Add your code here --------------------------------
  plt.plot(numComponents, errorTrain, 'r', label='TrainingSet')
  plt.plot(numComponents, errorValidation, 'b', label='ValidationSet')
  plt.plot(numComponents, errorTest, 'g', label='TestSet')
  plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure)
  plt.title('Classification error rates of different # of mixture components')
  plt.xlabel('# of mixture components')
  plt.ylabel('Classification error rates');
  plt.axis('tight')

  plt.draw()
  raw_input('Press Enter to continue.')

def computeErrorRate(logProb2, logProb3, target):
  incorr = 0
  for i in range(target.shape[1]):
    # print logProb2[i]
    # print target[0][i]
    if (logProb2[i] > logProb3[i] and target[0][i] == 1.0) or (logProb2[i] < logProb3[i] and target[0][i] == 0.0):
      incorr += 1
  errorRate = float(incorr)/target.shape[1]
  return errorRate

def q5():
  # Choose the best mixture of Gaussian classifier you have, compare this
  # mixture of Gaussian classifier with the neural network you implemented in
  # the last assignment.

  # Train neural network classifier. The number of hidden units should be
  # equal to the number of mixture components.

  # Show the error rate comparison.
  #-------------------- Add your code here --------------------------------
  inputs_train, inputs_valid, inputs_test, target_train, target_valid, target_test = LoadData('digits.npz')
  train2, valid2, test2, target_train2, target_valid2, target_test2 = LoadData('digits.npz', True, False)
  train3, valid3, test3, target_train3, target_valid3, target_test3 = LoadData('digits.npz', False, True)

  K = 2
  iters = 30
  minVary = 0.01
  p2, mu2, vary2, logProbX2 = mogEM(train2, K, iters, minVary)
  p3, mu3, vary3, logProbX3 = mogEM(train3, K, iters, minVary)

  logProb2_test = mogLogProb(p2, mu2, vary2, inputs_test)
  logProb3_test = mogLogProb(p3, mu3, vary3, inputs_test)
  errorRate_test_MoG = computeErrorRate(logProb2_test, logProb3_test, target_test)

  num_hiddens = 4
  eps = 0.2
  momentum = 0.0
  num_epochs = 1000
  W1, W2, b1, b2, train_error, valid_error, test_error_NN = TrainNN(num_hiddens, eps, momentum, num_epochs)
  print errorRate_test_MoG
  print test_error_NN
  ShowMeans(W1)

  raw_input('Press Enter to continue.')

if __name__ == '__main__':
  # q2()
  # q3()
  # q4()
  q5()

