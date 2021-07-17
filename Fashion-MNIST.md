# Fashion-MNIST

## 数据集介绍

```
"""Loads the Fashion-MNIST dataset.

This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories,
along with a test set of 10,000 images. This dataset can be used as
a drop-in replacement for MNIST. The class labels are:

| Label | Description |
|:-----:|-------------|
|   0   | T-shirt/top |
|   1   | Trouser     |
|   2   | Pullover    |
|   3   | Dress       |
|   4   | Coat        |
|   5   | Sandal      |
|   6   | Shirt       |
|   7   | Sneaker     |
|   8   | Bag         |
|   9   | Ankle boot  |

Returns:
    Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

    **x_train, x_test**: uint8 arrays of grayscale image data with shape
      (num_samples, 28, 28).

    **y_train, y_test**: uint8 arrays of labels (integers in range 0-9)
      with shape (num_samples,).

License:
    The copyright for Fashion-MNIST is held by Zalando SE.
    Fashion-MNIST is licensed under the [MIT license](
    https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

"""
```

## 数据集图片展示

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210717195735739.png#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210717195750946.png#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/2021071719580442.png#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/2021071719582248.png#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210717195846681.png#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210717195859856.png#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210717195910968.png#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210717195921264.png#pic_center)


其中其所对应的为一个一维数组，分别从一到十对应

```python
x_train_all=tf.constant(x_train_all)
y_train_all=tf.constant(y_train_all)
print(x_train_all.shape)
print(y_train_all.shape)
```

**运行结果：**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210717195941110.png#pic_center)


## 模型建立

### 模型一

**损失函数：交叉熵**

**优化器：梯度下降方法**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210717195953665.png#pic_center)


**模型如下图：**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210717200003717.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


### 模型二

这里借鉴了一篇论文论文里面推荐用VGG-11来训练效果会很好，会达到近91%正确率

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210717200016959.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20210717200027812.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzUxMzI0NjYy,size_16,color_FFFFFF,t_70#pic_center)


**VGG11模型**

```python
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU(inplace)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (7): ReLU(inplace)
    (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (9): ReLU(inplace)
    (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (12): ReLU(inplace)
    (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (14): ReLU(inplace)
    (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (17): ReLU(inplace)
    (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (19): ReLU(inplace)
    (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace)
    (2): Dropout(p=0.5)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace)
    (5): Dropout(p=0.5)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

```

#附注：这个模型建好了但是训练的巨慢，直接那种训练一天那样子，按照论文来说能达到91%的正确率，应该能吧

[这里先上传仓库了](https://github.com/hideonpython/Fashionminist)



## **BN层的作用**

Batch Normalization的作用
使用Batch Normalization，可以获得如下好处，

可以使用更大的学习率，训练过程更加稳定，极大提高了训练速度。
可以将bias置为0，因为Batch Normalization的Standardization过程会移除直流分量，所以不再需要bias。
对权重初始化不再敏感，通常权重采样自0均值某方差的高斯分布，以往对高斯分布的方差设置十分重要，有了Batch
Normalization后，对与同一个输出节点相连的权重进行放缩，其标准差σ也会放缩同样的倍数，相除抵消。
对权重的尺度不再敏感，理由同上，尺度统一由γ参数控制，在训练中决定。
深层网络可以使用sigmoid和tanh了，理由同上，BN抑制了梯度消失。
Batch Normalization具有某种正则作用，不需要太依赖dropout，减少过拟合。

[这里有一个很好的博客](https://blog.csdn.net/weixin_44023658/article/details/105844861)