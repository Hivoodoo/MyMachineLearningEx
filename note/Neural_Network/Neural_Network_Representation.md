#非线性分类器(Non-linear hypotheses)
##为什么使用非线性分类器
我们举几个栗子:
![house](file:\\\./image/exampleHouse.jpg)
假如我们有一个数据空间如左上角坐标系所示,那么我们要的模型需要如右边公式所示的预测函数.
假设有n个特征那么计算二次多项式就有O(n^2)的复杂度.n能有多大?我们来看下面这个栗子.
![Car](file:\\\./image/exampleCar.jpg)
假设我们需要识别汽车,假如选取图像上两个点,那么就如左边坐标系所示,这没什么.
但实际上我们需要的数据空间时整张图片所有的像素.也就是假设图像是$50 * 50$那么我们就有$2500$个像素点.也就是需要2500个特征.
刚才说的是灰度图,如果时RGB的话时$7500$个,实际运用中也不只$50 * 50$ 这么小.
综上所述,线性分类器肯定是不行的.

#神经网络
神经网络是模拟人类的神经元时发明的.
![TrueNeural](file:\\\./image/TrueNeural.jpg)
##神经网络的表示
![SingalNeural](file:\\\./image/SingleNeural.jpg)
一个神经元(Neuron)就是如图所示,他和我们的逻辑回归表示并没有什么区别.
Sigmoid Function叫做激励(activation)函数.
$\theta$也被成为权重(weights)
![network1](file:\\\./image/network1.jpg)
神经网络由多个神经元连接而成,其中第一层被称为输入层(input layer);最后一层被称为输出层(output layer);其他的被称为隐藏层(Hidden layer).
###记号
![mark](file:\\\./image/mark.jpg)
- $a _ {i} ^{(j)} $:
第j层第i个激励
- $\Theta^{(j)}$(波矩阵?)
第 j 层到第 j+1 层的参数控制映射

我们通常使用$\Theta^{(j)}$矩阵来表示参数,表示方法如上所述.其中$\Theta$的大小为$(s _ {j+1} * (s _ j + 1)$,$s _ j$ 为第j层激励的个数.

##向量化(Vectorized)实现
![Vectorized](file:\\\./image/Vectorized.jpg)
我们可以将通过向量化来简化神经网络的计算.
我们可以把左下角图中Layer 2的Sigmoid函数提出,这样我们可以用$\Theta$向量的形式表示出整个Layer 2.
也就是右边所示的过程.
写成一般形式就是:
$$
Add\quad a _ 0^{(i-1)}=1\\
z^{(i)}=\Theta^{(i-1)}a^{(i-1)}\\
a^{(i)}=g(z^{(i)})\\
$$
我们循环执行这个步骤.直到输出层输出$h _ \Theta (x)$,这就是**前向传播(Forward propagation)**.
##架构(Architecture)
![Architecture](file:\\\./image/Architecture.jpg)
我们把神经元连接的方式叫做架构.

##神经网络栗子
![AndFunction](file:\\\./image/AndFunction1.jpg)
为了方便我们理解神经网络,这里有几个栗子.
假如我们有右边的数据空间,那么比较好的边界就是如图所示.
我们先简化数据为左边的图,也就是说我们需要训练一个模型求得左下的y式,很复杂.
![AndFunction](file:\\\./image/AndFunction2.jpg)
我们先来个简单的AND运算.如上图,假设我们的theta如下
$$ \theta  = \begin{bmatrix}
-30\\\\
+20\\\\
+20\\
\end{bmatrix}
$$
那么就能预测出如真值表的值.
那么同理,我们就能训练出能求或与非的模型.那么我们要求XNOR运算怎么办?
![AndFunction](file:\\\./image/AndFunction3.jpg)
这里我们来构建一个神经网络,每个节点训练出不同的模型,达到不同的效果,最后达到输出层.
##多类别分类
之前说过多类别分类问题采用(one vs all)
![MC](file:\\\./image/MC1.jpg)
假设我们有4个类别需要识别,那么我们就有四个分类器.
![MC](file:\\\./image/MC2.jpg)
每个分类器的含义同之前,$P(y=j|\theta^{(i)} _ {j})$.

##代价函数
---

**Tips:**
数据空间=$$
\{ (x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),\cdots (x^{(m)},y^{(m)}) \}
$$
$L = 网络层数$
$S _ l = 第l层神经元数.$
---

在神经网络中,神经单元的代价函数就是逻辑回归中的代价函数的一般式.
$$
h _ \Theta(x) \in \mathbb{R}^K\qquad
(h _ \Theta(x)) _ i=i^{th} \text{output}
$$
$$
\begin{aligned}
J(\Theta) = & -\frac {1}{m} \begin{bmatrix}
\sum _ {i=1} ^ m\sum _ {k=1} ^ K
 y _ k ^ {(i)}log(h _ \Theta(x^{(i)})) _ k+(1-y _ k ^{(i)})log(1-(h _ \Theta(x^{(i)})) _ k)
\end{bmatrix}\\
&+\frac{\lambda}{2m}
\sum _ {l=1} ^ {L-1}
\sum _ {i=1} ^ {s _ l}
\sum _ {j=1} ^ {s _ l+1}
(\Theta _ {ji} ^ {(l)})^2
\end{aligned}
$$


解释一下:
$h _ \Theta(x) \in \mathbb{R}^K$就是神经网络的输出在K维向量中,二元分类K就是1
$(h _ \Theta(x)) _ i$表示第$i$个输出.
$y _ k ^ {(i)}log(h _ \Theta(x^{(i)})) _ k$中$y$是一个K维向量,$y _ k$就是第K维向量,如果这组样本属于第K组,那么$y _ k$就是1,vise varsa.
需要注意的是$\sum _ {j=1} ^ {s _ l+1}$中依旧可以遵循"不把偏差项$\Theta _ {i0} ^{(l)}\ (j = 0)$正规化".

##反向传播算法(Backpropagation Algorithm)
我们的目标是:$$\min _ \Theta J(\Theta)$$
为了找到这个$\Theta$,我们需要计算:

- $J(\Theta)$
- $\frac{\partial}{\partial\Theta _ {ij} ^ {(l)}} J( \Theta)$
其中$J(\Theta)$我们已经知道了怎么计算.
那么我们需要计算的就是$J(\Theta)$的偏导,这时候我们用到反向传播算法:
直观上来说就是对每一个节点计算$\delta _ j ^ {(l)}$(第l层第j个节点的误差).
假设我们有一个网络如图:
![delta](file:\\\./image/whatdelta.jpg)
首先我们从输出节点开始求$\delta^{(4)}$
$\delta _ j ^{(4)} = a _  j ^{(4)} - y _ j$.
变成向量化的形式:$\delta^{(4)} = a^{(4)} - y$
然后往前倒推:
$\delta^{(l)} = (\Theta^{(l)})^T\delta^{(l+1)}. * g'(z^{(l)})$,其中$g'(z^{(l)})=a^{(l)}. * (1-a^{(l)})$.
直到$l=1$,输入层不需要计算误差.
那么忽略$\lambda$,需求偏导公式可变为:
$\frac{\partial}{\partial\Theta _ {ij} ^ {(l)}} J( \Theta)=a _ j ^ {(l)} \delta _ i ^{(l+1)}$
###实现
![BPimplement](file:\\\./image/BPimplement.jpg)
使用上述公式$\frac{\partial}{\partial\Theta _ {ij} ^ {(l)}} J( \Theta)=a _ j ^ {(l)} \delta _ i ^{(l+1)}$需要使用$a$和$\delta$.
我们引入$\Delta$,$\Delta _ {ij} ^{(l)}$是用于计算$\frac{\partial}{\partial\Theta _ {ij} ^ {(l)}} J( \Theta)$的一个累加项.
对于每一组样例,
我们先用FP来计算$a$,再反向计算误差$\delta^{(l)}$
最后按照偏导项公式累加$\Delta$,得到公式:
$$\frac{\partial}{\partial\Theta _ {ij} ^ {(l)}} J( \Theta)=D _ {ij} ^{(l)} = \frac{1}{m}\Delta _ {ij}^{(l)} + \lambda\Theta _ {(ij)} ^{(l)}$$
若将$\Delta$向量化得到$\Delta^{(l)}:=\Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$.
##矩阵向量化(Matrices "Unroll" into vectors)
![vectorized](file:\\\./image/vectoried.jpg)
fminunc这些高级函数在使用的时候需要传递向量,而$\Theta$这种却是矩阵.
我们可以把矩阵展开,来达到向量化的目的,然后在costFunction中重组$\Theta$等矩阵.
![Unroll](file:\\\./image/Unroll.jpg)
##梯度检测(Gradient Checking)
简单的说就是使用双侧差分(two sided difference)求近似导数.
![gradApprox](file:\\\./image/gradApprox.jpg)
在神经网络中具体实现就是如下.
![gradApprox2](file:\\\./image/gradApprox2.jpg)
一般我们取$\epsilon\approx 10^{-3}$,然后检验上述所得向量(gradApprox)近似于DVec.
总结:
1. BP求DVec.
2. 求gradApprox.
3. 检测是否相似
4. 关闭检测(因为梯度检测很慢),使用BP学习.
##随机初始化(Random Initialization)
经过之前的学习,我们发现每一层的所有单元都会经过相同的训练.所以我们使用随机初始化来打破这种对称性.
随机初始化指的是随机化初始的参数矩阵,使他们接近0却不完全相同.

#总结
##架构(Architecture)
![Sumary](file:\\\./image/Sumary.jpg)
- 输入单元数:特征空间维度.
- 输出单元数:类数
- 隐藏层:通常为一层,每层激励个数相同,数量选择时需要考虑输入输出层,通常稍大于输入层.
##训练
1. 随机初始化权重.
2. FP求$h _ \Theta$,也就是$a$.
3. 计算代价函数$J(\Theta)$
4. BP求 $\frac{\partial}{\partial\Theta _ {ij} ^ {(l)}} J( \Theta)$.
5. 使用梯度检测比较BP所求$\frac{\partial}{\partial\Theta _ {ij} ^ {(l)}} J( \Theta)$,然后关闭梯度检测.
6. 使用梯度下降等算法和BP求$\Theta$($J(\Theta)$是非凸函数).


#神经网络背景知识
##起源
人们想要模拟大脑,因为大脑是最好的学习模型.
兴起与80s~90s,但随后衰退.但是今年由于数据量和计算速度的提高又变得兴起
##思考
人类的大脑能学习很多的东西,比如分辨事物,学习,计算,说话等等.
我们也能写很多算法来模拟这些"任务",实现该过程.但是大脑的学习应该是只有一个学习算法的(The "one single learning algorithm" hypothesis).
##实验(Neural Re-wired Experiment)
![experiment](file:\\\./image/AnimalExp.jpg)
科学家把动物视神经切断,而把听觉神经连接到本来由视神经连接的位置(视觉皮层),最后发现动物能完成视觉辨别任务.
![rewiredusing](file:\\\./image/rewirdUsing)
左上角:在额头上佩戴灰度摄像机,数据输出到舌头的电极,刺激舌头.失明的人能在几十分钟内学会"看".
右上角:人类声呐定位.通过打响指或者咂舌来制造声音,通过听觉来分辨回声,定位周围的物体(这是一种训练,手机上玩过一个游戏叫 Echo还是Dark Echo的就是这种感觉).
左下角:蜂鸣腰带,朝向北时腰带蜂鸣很强,使人类拥有鸟类的方向感(这个..恩..).
右下角:给青蛙按第三只眼睛,青蛙能学着使用(卧槽,那岂不是给人类大脑接一个高清摄像头就行,学习量可能有点大吧)
