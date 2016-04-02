(Title:)【HowTo ML】正则化

#过拟合(Overfitting)与欠拟合(Underfitting)
![fitting](file:\\\./image/Fitting.jpg)
##欠拟合
一个模型不能很好的拟合数据,或者说有很强的偏向,或者说有很大的偏差(High bias)
此时我们不能很好的预测新样本数据
##过拟合
特点:$J(\theta)\approx 0
一个模型过于拟合训练数据,或者说有很高的方差(High varionce).
这种情况可能也会造成函数过大,变量过多的情况
##小结
我们需要更为准确的预测,一组数据不可能代表所有的可能性.
所以我们需要泛化(generalize)模型来预测新的样本数据.
###解决
1. 减少特征
	- 人工删除
	- 模型选择算法
2. 正则化(Regularization)
	- 保留所有特征但是减少$\theta _ J$的数量级或者大小.
	- 其中每一个都为预测$y$贡献一点点

#代价函数
对于正则化来,说我们通过改变代价函数来约束参数的大小,从而改变该特征对模型影响的大小来使模型更为简单.举个栗子:
![example](file:\\\./image/costFunction)
在这个栗子中,由于costFunction,$\theta _ 3$ 和 $\theta _ 4$将会约等于0,也就是贡献非常小.
我们将这种代价函数一般话:
$$
J(\theta) = \frac {1}{2m}
\begin{bmatrix}
\sum _ {i=1}^{m}(h _ \theta (x^{(i)}) - y^{(i)}) +
\lambda\sum _ {j = 1} ^{n} \theta _  {j} ^{2}
\end{bmatrix}
$$
首先这个代价函数有两个目标:
1. 匹配训练集
2. 保持参数尽可能小

其中$\lambda\sum _ {j = 1} ^{n} \theta _  {j} ^{2}$叫做正规化参数(regularization parameter)也就是我们常说的模型的偏见,用于第二个目标
$\lambda$用于平衡两个目标之间的平衡.
如果$\lambda$过大会欠拟合,过小会过拟合.
**注意:**通常不会正规化$theta _ 0$,虽然影响不大.

#梯度下降
那么通过上面代价函数的修改,梯度下降则变为:
$$
\begin{array}{l}
repeat\;until \; convergence \{ \\
\qquad \theta _ 0 := \theta _ 0 - \alpha\frac{1}{m} \sum^m _ {i=1}( h _ \theta(x^{(i)})-y^{(i)})x _ 0^{(i)} \\
\qquad \theta _ j := \theta _ j(1-\alpha\frac{\lambda}{m}) - \alpha\frac{1}{m} \sum^m _ {i=1}( h _ \theta(x^{(i)})-y^{(i)})x _ j^{(i)} 
\qquad(j = 1,2,...,n)
\\
\}
\end{array}
$$
一般的$(1-\alpha\frac{\lambda}{m}) < 1$,一般小于1一点点.

#正规方程
我们在线性回归中介绍了正规方程.
在这里我们也可以用来计算正规方程参数.
下面是线性回归中无偏向的正规方程:
$$X(design\ matrix) = \left[ \begin{matrix}
(x^{(1)})^T&\\
\vdots&\\
(x^{(n)})^T&\\
\end{matrix}
\right]
$$

$$\theta=(X^TX)^-1X^Ty$$

那么加上偏好
$$
\theta=
(X^TX + \lambda
\begin{bmatrix}
0&0&0&\cdots&0\\
0&1&0&\cdots&0\\
0&0&1&\cdots&0\\
\vdots&\vdots&\vdots&\ddots\\
0&0&0&&1
\end{bmatrix}
 )^-1
X^Ty
$$
其中矩阵大小为$(n+1) * (n+1)$

##可逆矩阵
可以证明:只要$\lambda > 0$那么存在上述矩阵.
