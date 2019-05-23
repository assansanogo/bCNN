import matplotlib
matplotlib.use('Agg')

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

    print train_images.shape
    print train_labels.shape
    print test_images.shape
    print test_labels.shape



if __name__=='main':
    main()