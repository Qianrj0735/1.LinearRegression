#coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = [[21809.8,289847.7,311485.13,111.3,400.2,426.2
],[19109.4,266621.5,251683.77,102.9,377.5,406.3
],[17174.7,220001.51,224598.77,97.5,357.8,394.1
],[15780.8,166217.13,172828.4,120.3,378.2,398.9
],[13785.8,152560.08,137323.94,107.7,353.8,376.7
],[11759.5,126035.13,109998.16,101.5,343.2,362.9
],[10493,107278.8,88773.61,108.3,333.2,359.3
],[9421.6,95969.7,70477.43,110.6,317.6,356.4
],[8472.2,84118.57,55566.61,101.4,299.3,346.7
],[7702.8,70881.79,43499.9,100.5,292.6,347
],[6859.6,59871.59,37213.5,99.1,299.2,351.6
],[6280,53147.2,32917.7,99.1,303.1,354.4
],[5854,45837.3,29854.7,95.8,294.8,359.8
]]

Y_ = [565.00,536.10,519.00,522.70,493.60,471.00,464.00,455.80,438.70,433.50,437.00,434.00,432.20]

X=np.array(X)
Y_=np.array(Y_)


Xmin, Xmax = X.min(), X.max() # 求最大最小值
X = (X-Xmin)/(Xmax-Xmin) # (矩阵元素-最小值)/(最大值-最小值)

Ymin, Ymax = Y_.min(), Y_.max() # 求最大最小值
Y_ = (Y_-Ymin)/(Ymax-Ymin) # (矩阵元素-最小值)/(最大值-最小值)

print ("X:\n",X)
print ("Y_:\n",Y_)

x = tf.placeholder(tf.float32, shape=(1,6))
x1 = tf.placeholder(tf.float32, shape=(1,6))
y_= tf.placeholder(tf.float32, shape=(1))
w = tf.Variable(tf.random_normal([6,1], mean=0,stddev=0.3, seed=12345))
b= tf.Variable(tf.random_normal([1], mean = 0.5,stddev=0.3, seed=54321))

a = tf.matmul(x, w)
y = tf.add(a,b)
loss_mse = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print ("训练前w:\n", sess.run(w))
    print ("训练前 b:  ", sess.run(b))
    #print ("b:\n", sess.run(b))
    print ("\n")
    num = 1

    for a in range(19499):
        i = a%13
        i1 = i+1
        sess.run(train_step, feed_dict={x: X[i:i1 , :], y_: Y_[i:i1]})
        if a%650 == 0:
            yp = []
            ls = []
            yp = np.array(yp)
            ls = np.array(ls)
            dateOne = np.zeros(13)
            for c in range(13):
                i = c%13
                i1 = i+1
                dateOne[i] = i
                ypredicted = sess.run(y, feed_dict={x: X[i:i1 , :], y_: Y_[i:i1]})
                ypredicted = (ypredicted*(Ymax-Ymin))+Ymin
                yp=np.append(yp,ypredicted)
                loss = sess.run(loss_mse, feed_dict={x: X[i:i1 , :], y_: Y_[i:i1]})
                ls = np.append(ls,loss)
            plt.subplot(3, 10, num)
            plt.plot(dateOne,(ls*1000)+400,color='green',linewidth=1.0 )
            plt.scatter(dateOne,yp,c = 'r',marker = 'o' )
            plt.plot(dateOne,(Y_*(Ymax-Ymin))+Ymin,color='blue',linewidth=1.0 )
            num+=1
        plt.plot
    plt.show()
    print ("训练后 w:\n", sess.run(w))
    print ("训练后 b:  ",sess.run(b))
    print ("\n")