import matplotlib
#matplotlib.use('Agg')

import pandas
from bigdl.dataset import mnist
from bigdl.util.common import *
import matplotlib.pyplot as plt
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



def get_dataset(sc, datum):
    (train_images, train_labels) = datum[0]
    (test_images, test_labels) = datum[1]

    training_mean = np.mean(train_images)
    training_std = np.std(train_images)
    rdd_train_images = sc.parallelize(train_images)
    rdd_train_labels = sc.parallelize(train_labels)
    rdd_test_images = sc.parallelize(test_images)
    rdd_test_labels = sc.parallelize(test_labels)

    rdd_train_sample = rdd_train_images.zip(rdd_train_labels).map(lambda (features, label):
                                                                  Sample.from_ndarray(
                                                                      (features - training_mean) / training_std,
                                                                      label + 1))
    rdd_test_sample = rdd_test_images.zip(rdd_test_labels).map(lambda (features, label):
                                                               Sample.from_ndarray(
                                                                   (features - training_mean) / training_std,
                                                                   label + 1))



if __name__=='__main__':
    main()