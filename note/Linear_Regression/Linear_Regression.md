
tips:

- $(x,y)$代表样例集。
- $(x^{(1)},y^{(1)})$为第一组样例。
- function hypothesis: 假设函数
	- 表示: 
	- $h _ \theta(x) = \theta _ 0+\theta _ 1 * x$
	- $\theta _ i$: Parameter 参数

#监督学习(Supervised Learning)
给出的数据具有某种标记
对每个数据给出"正确答案" 
##回归问题(Regression Problem)
术语上回归意味着要预测这类连续值属性的种类。

###代价函数
####平方误差函数
$$J(\theta _ 0,\theta _ 1) = \frac{1}{2}\sum _ {i=1}^{m}(h _ \theta(x^{(i)})-y^{(i)})^2$$

contour plots or contour figures:轮廓图
视觉化的参数估计

回归的目标：
$$argmin(\theta _ 0,\theta _ 1) J(\theta _ 0,\theta _ 1)$$
意义是使得J函数最小化的参数

###多源线性回归(Multivariate Linear Regression)
也就是多个设置多个参数

$h(x)=\theta^TX$
$x = [x _ 0 x _ 1 x _ 2 \cdots x _ n] , x _ 0 = 0$ $\theta = [\theta _ 0 \theta _ 1\theta _ 2 \cdots \theta _ 3]$


###梯度下降(Gradient descent algorithm)
hint:

- $:=$	赋值
- $\alpha$	学习速率(learning rate)
- 梯度下降中需要同时更新 $\theta _ 0$ 和$\theta _ 1$

梯度下降算法伪代码:
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

####梯度下降的技巧
#####**特征缩放(Feature Scaling)**
思想：使特征处于相似的值将加快缩放
一般来说将特征缩放至 $x _ i \in [-3,3] $都行(经验)
#####**均值归一化(mean normalization)**
(我的理解是带偏移量的缩放,使$x _ i \in [-1,1]$)
使用 $x _ i$ 来替代 $x _ i-\mu _ i$ 来使特征接近0 (除了$x _ 0$定义为$1$）
一般公式：$x _ i <- (x _ i-\mu _ i)/s _ i$
$\mu _ i$	一个任意的 $x _ i$值
$s _ i$	,$x _ i$的范围($max-min$)
#####**梯度下降率(Gradient decent)**
选取过大的下降率会使$J(\theta)$不下降、上升或者跳跃
选取过小的下降率会使$J(\theta)$下降过慢

###多项式回归(Polynomial regression)
**normalization is import!**
####正规方程(Normal Equation)
$$X(design\ matrix) = \left[ \begin{matrix}
(x^{(1)})^T&\\
\vdots&\\
(x^{(n)})^T&\\
\end{matrix}
\right]
$$

$$\theta=(X^TX)^-1X^Ty$$
特点：
需要算逆矩阵($O(n^3)$)

出现$X^TX$不可逆(Non-invertible)的情况:

- 冗余特征(Redundant feature/linearly dependent):
	一个特征的值可以由另一些变量来表示
- 特征过多:
	e.g. $m<=n$
	解决:
	- 删除特征
	- 正则化
