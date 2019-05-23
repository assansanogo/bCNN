import matplotlib
#matplotlib.use('Agg')

import pandas
from bigdl.dataset import mnist
from bigdl.util.common import *
import matplotlib.pyplot as plt
from pyspark import SparkContext
from keras.models import Model

from bigdl.nn.layer import *
from bigdl.nn.criterion import *
from bigdl.optim.optimizer import *
from bigdl.util.common import *
from bigdl.dataset.transformer import *
from bigdl.dataset import mnist
from pyspark import SparkContext
from matplotlib.pyplot import imshow



def main():
    sc = SparkContext.getOrCreate(conf=create_spark_conf().setMaster("local[*]").set("spark.driver.memory", "2g"))
    init_engine()
    mnist_path = "datasets/mnist"
    (train_images, train_labels) = mnist.read_data_sets(mnist_path, "train")
    (test_images, test_labels) = mnist.read_data_sets(mnist_path, "test")
    datum = []

    datum.append((train_images, train_labels))
    datum.append((test_images, test_labels))
    print train_images.shape
    print train_labels.shape
    print test_images.shape
    print test_labels.shape

    get_dataset(sc, datum)

def get_model(class_num =10):
    # Initialize a sequential container
    model = Sequential()
    # encoder
    model.add(Reshape([1, 28, 28]))
    model.add(SpatialConvolution(1, 6, 5, 5).set_name('conv1'))
    model.add(ReLU())
    model.add(SpatialMaxPooling(2, 2, 2, 2).set_name('pool1'))
    model.add(SpatialConvolution(6, 12, 5, 5).set_name('conv2'))
    model.add(ReLU())
    model.add(SpatialMaxPooling(2, 2, 2, 2).set_name('pool2'))
    model.add(Reshape([12 * 4 * 4]))
    model.add(Linear(12 * 4 * 4, 100).set_name('fc1'))
    model.add(Linear(100, class_num).set_name('score'))
    model.add(LogSoftMax())
    return model



def get_dataset(sc, datum):
    (train_images, train_labels) = datum[0]
    (test_images, test_labels) = datum[1]

    training_mean = np.mean(train_images)
    training_std = np.std(train_images)

    rdd_train_images = sc.parallelize(train_images,  numSlices=1000)
    rdd_train_labels = sc.parallelize(train_labels, numSlices=1000)

    rdd_test_images = sc.parallelize(test_images, numSlices=1000)
    rdd_test_labels = sc.parallelize(test_labels, numSlices=1000)

    rdd_train_sample = rdd_train_images.zip(rdd_train_labels).map(lambda (features, label):
                                                                  Sample.from_ndarray(
                                                                      (features - training_mean) / training_std,
                                                                      label + 1))

    #print(rdd_train_sample.take(10))
    rdd_test_sample = rdd_test_images.zip(rdd_test_labels).map(lambda (features, label):
                                                               Sample.from_ndarray(
                                                                   (features - training_mean) / training_std,
                                                                   label + 1))

    return([rdd_train_images, rdd_test_sample])


def main2():
    mods = get_model()


if __name__=='__main__':
    main2()