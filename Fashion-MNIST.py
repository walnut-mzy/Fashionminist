import tensorflow as tf
from tensorflow import keras
from PIL import Image
from tensorflow.keras import losses, optimizers
def show_pic(num,str1=None):
    if num<=0:
        return "error number"
    # 画出第一张图片
    im = Image.fromarray(x_train_all[num-1])
    if str1:
        c=im.save(str(str1)+".jpeg")
    im.show()


if __name__ == '__main__':
    batch=640
    # 导入Fashion-mnist数据集
    fashion_mnist = keras.datasets.fashion_mnist
    print(fashion_mnist)
    # #拆分训练集和测试集
    (x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
    # #显示十张图片图片
    # for i in range(1,10):
    #     show_pic(i)
    x_train_all = tf.constant(x_train_all/255)
    x_train_all=tf.expand_dims(x_train_all, 1)
    x_train_all=tf.expand_dims(x_train_all, 1)
    print(x_train_all.shape)
    y_train_all=tf.one_hot(y_train_all, depth=10)
    y_train_all = tf.expand_dims(y_train_all, 1)
    print(y_train_all.shape)
    # 构建训练datesets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_all, y_train_all))
    print(train_dataset)
    # 对训练集进行处理
    train_dataset.shuffle(buffer_size=10000).batch(batch)

    #构建测试集
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    criteon = losses.categorical_crossentropy
    optimizer = optimizers.Adam()
    # model = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.Flatten(input_shape=(28,28)),  # Flatten函数的作用是将输入的二维数组进行展开，使其变成一维的数组
    #         tf.keras.layers.Dense(256, activation='relu'),  # 创建权连接层，激活函数使用relu
    #         tf.keras.layers.Dropout(0.2),  # 使用dropout缓解过拟合的发生
    #         tf.keras.layers.Dense(10, activation='softmax'), # 输出
    #        # tf.keras.layers.Reshape(),
    #     ]
    # )
    #使用VGG11来训练
    model=tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),input_shape=(None,28,28),strides=(1,1),activation="relu",padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=2,padding="same"),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation="relu",padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,padding="same"),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation="relu",padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,padding="same"),
        tf.keras.layers.BatchNormalization(), #BN层在vgg里面是没有的但是写论文的人用了那咱就使用
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu",padding="same"),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation="relu",padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,padding="same"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation="relu",padding="same"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation="relu",padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation="relu",padding="same"),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation="relu",padding="same"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2,padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Reshape((512,)),
        tf.keras.layers.Dense(4096,activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10,activation="softmax")
    ])
    model.summary()
    model.compile(optimizer=optimizer,
                  loss=criteon,
                  metrics=['accuracy'])
    model.fit(train_dataset,batch_size=batch, epochs=100)