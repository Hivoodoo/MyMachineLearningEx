#分类问题(Classification Problem)
样本因变量$Y$被标记为种类,也就是说分类是要预测一个离散值的输出。对应的问题就是分为 $Y$ 类的概率。
支持向量机支持无数多的属性

##逻辑回归(Logistic Regression)
逻辑回归:根据自变量确定因变量的种类,$h _ \theta(x) \in [0,1] $ 但是在分类问题中$h _ \theta(x)$可以$>1$或者$<0$.
逻辑回归通常使用的是二元,这类问题中 $y \in \{0,1 \} $.
其中:

	0: Negative Class
	1: Positive Class

常见问题:垃圾邮件过滤,欺诈检测,肿瘤识别.

通常我们不会使用线性回归来解决分类问题.
![原因](file\\\:./image/badlinear.jpg)
对于两种分类来说,我们通常会引入阀值(Threshold),但当我们加入一个$X$较大的数据通常会对阀值造成很大影响.
但现实中影响却不大.

###假设函数(Hypothesis Representation)
为了使$h _ \theta(x) \in [0,1]$,我们需要更改回归问题时用到的函数,变成形式$h _ \theta(x) = g(\theta^TX)$.
其中$g$称为S型函数(Sigmoid function)或者(Logistic Function),定义为$g(z) = \frac {1}{1+e^{-z}}$,函数图像如下:
![g(z)](file\\\:./immage/LGG.png)
带入可得
$$h _ \theta(x) = \frac {1}{1+e^{-\theta^Tx}}$$
我们也能将$h _ \theta(x)$解释为如下形式:
$$h _ \theta(x) = P(y=1|x;\theta)$$
解释为:当给定$x$,概率参数为$\theta$时,$y=1$的概率.
对于二元分类问题来说$y=0$和$y=1$是对立事件,所以$y=0$的概率就是$1-P(y=1|x;\theta)$.

###决策边界(Decision boundary)
决策边界是决策函数的一个属性,由参数决定,能根据参数将样本分为不同类别的边界.
![决策边界](file\\\:./image/DB.png)
为了更好的理解决策边界,现在我们考虑逻辑函数$g(z)$的对称性.通常,我们会将阀值设定在0.5,也就是说当$h _ \theta(x) \ge 0.5$时$y=1$.
其实,也就是当$h _ \theta(x) \ge 0$时 $y=1$.

###代价函数(Cost Function)
对于线性回归来说:
$$
\begin{align}
J(\theta) & = \frac {1}{m} \sum _ {i = 1} ^ { m } \frac {1} {2}(h _ \theta ( x^{(i)} ) - y ^ {(i)})^2 \\
& = \frac {1}{m} \sum _ {i = 1} ^ { m } Cost(h _ \theta ( x^{(i)} ), y ^ {(i)})
\end{align}
$$
代价函数可以方便的表示为:
$$
Cost(h _ \theta ( x ), y ) = \frac {1} {2} (h _ \theta ( x ) , y)
$$

我们为什么不能直接套用线性回归的代价函数?因为我们期望代价函数能得到一个最小的值,但线性回归的代价函数是非凸函数(Non-convex),如左图,有多个局部最小值.我们期望能得到右图.
![函数图像](file\\\:./image/LRCF.jpg)

那么逻辑回归的代价函数应该设计成什么样呢?
$$
Cost(h _ \theta (x) , y) =  \big\lbrace
\begin{array}
 -log(h _ \theta(x)&(if\ y=1) \\
 -log(1 - h _ \theta(x))&(if\ y=0) 
\end{array}
\big\rbrace
$$
![-log(x)的函数图像](file\\\:./image/LoCF.jpg)
![-log(1-x)的函数图像](file\\\:./image/LoCF1.jpg)
假设有个目标的$x \to 0$,我们预测为1,但事实上却相反,那这个算法就会收到较大的惩罚.反之亦然.
但是有处于中部的预测错误,我们就不会受到较大惩罚.

为了方便编写,我们可将$Cost$函数转换成:
$$
Cost( h _ \theta (x), y ) = 
-y\ log(h _ \theta) - (1-y)log(1-h _ \theta(x))
$$
所以
$$
\begin{align}
J(\theta) 
& = \frac {1}{m} \sum _ {i = 1} ^ { m } Cost(h _ \theta ( x^{(i)} ), y ^ {(i)})\\
& = - \frac {1}{m} \sum _ {i=1} ^ {m} y\ log(h _ \theta) + (1-y)log(1-h _ \theta(x))
\end{align}
$$

###逻辑回归的梯度下降
梯度下降算法:
$$
\begin{array}{l}
repeat\;until \; convergence \{ \\
\qquad
\theta _ j := \theta _ j - \alpha \frac{\partial}{\partial\theta _ j}J(\theta _ 0,\theta _ 1) (for\ j : n,n=2) \\
\}
\end{array}
$$
带入待解函数后可化简为
$$
\begin{array}{l}
repeat\;until \; convergence \{ \\
\qquad \theta _ j := \theta _ j - \alpha\frac{1}{m} \sum^m _ {i=1}( h _ \theta(x^{(i)})-y^{(i)})x _ j^{(i)} (for\ j : n) \\
\}
\end{array}
$$
惊奇(?)的发现,里面的公式其实没有变!
其实改变的只有$ h _ \theta(x) $函数.
###特征缩放
线性回归的特征缩放是有效的!

###其他参数求取方法

- Conjugate gradient
- 共轭梯度法BFGS(变尺度法)
- L-BFGS(限制变尺度法)

优点:

- 不需要手动选择学习率$\alpha$(通过线性搜索选择)
- 比梯度下降快

Octave中使用函数:
fminunc(无约束最小化函数):
```matlab
 -- Function File: fminunc (FCN, X0)
 -- Function File: fminunc (FCN, X0, OPTIONS)
 -- Function File: [X, FVAL, INFO, OUTPUT, GRAD, HESS] = fminunc (FCN,
          ...)Octave:
```
具体用法
```matlab:
##定义代价函数
function [jVal, gradient] = costFunction(theta)
	#...
options = optimset( 'GradObj', 'on', 'MaxIter', '100');

initialTheta=zeros(2,1);
[optTheta, functionVal, exitFlag] ...
	= fminunc(@costFunction, initalTheta, option;

```
options 解释:

1. 梯度下降
2. 打开
3. 最大迭代次数
4. 100次

exitFlag:
如果为1就是收敛了,具体查阅手册.

###解决多类别分类问题(Multi-class classification)
思想: 把多类别分类转化为多次逻辑回归,也就是每次分为两类,一类为本次所选取的类,另一类为剩下的(One-vs-Rest).
![转化](file:\\\./image/MCC.jpg)
最后我们能得到K(种类个数)个逻辑分类器.也就可以得到每一种可能性的概率了,通常我们取最大的那个作为结果.
