import numpy as np
import tensorflow as tf

import tf_cka


class CKA_Test(tf.test.TestCase):

    def setUp(self):
        super(CKA_Test, self).setUp()
        self.BATCH_SIZE = 128
        self.DIM = 512
        self.X = np.random.randn(self.BATCH_SIZE, self.DIM)
        self.Y = np.random.randn(self.BATCH_SIZE, self.DIM)

    def testLinearKernel(self):
        np_result = tf_cka.linear_kernel(self.X)
        tf_result = tf_cka.linear_kernel_tf(self.X)

        self.assertAllClose(np_result, tf_result.numpy())

    def testCentering(self):
        np_result = tf_cka.centering(tf_cka.linear_kernel(self.X))
        tf_result = tf_cka.centering_tf(tf_cka.linear_kernel_tf(self.X))

        self.assertAllClose(np_result, tf_result.numpy())

    def testHSIC(self):
        np_result = tf_cka.hsic(self.X, self.Y, kernel=tf_cka.linear_kernel)
        tf_result = tf_cka.hsic_tf(self.X, self.Y, kernel=tf_cka.linear_kernel_tf)

        self.assertAllClose(np_result, tf_result.numpy())

    def testCKA(self):
        np_result = tf_cka.cka(self.X, self.Y, kernel=tf_cka.linear_kernel)
        tf_result = tf_cka.cka_tf(self.X, self.Y, kernel=tf_cka.linear_kernel_tf)

        self.assertAllClose(np_result, tf_result.numpy())


if __name__ == '__main__':
    tf.test.main()

