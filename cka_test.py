import numpy as np
import tensorflow as tf

import cka

class CKA_Test(tf.test.TestCase):

    def setUp(self):
        super(CKA_Test, self).setUp()
        self.BATCH_SIZE = 128
        self.DIM = 512
        self.X = np.random.randn(self.BATCH_SIZE, self.DIM)
        self.Y = np.random.randn(self.BATCH_SIZE, self.DIM)

    def testLinearKernel(self):
        np_result = cka.linear_kernel(self.X)
        tf_result = cka.linear_kernel_tf(self.X)

        self.assertAllClose(np_result, tf_result.numpy())

    def testCentering(self):
        np_result = cka.centering(cka.linear_kernel(self.X))
        tf_result = cka.centering_tf(cka.linear_kernel_tf(self.X))

        self.assertAllClose(np_result, tf_result.numpy())

    def testHSIC(self):
        np_result = cka.hsic(self.X, self.Y, kernel=cka.linear_kernel)
        tf_result = cka.hsic_tf(self.X, self.Y, kernel=cka.linear_kernel_tf)

        self.assertAllClose(np_result, tf_result.numpy())

    def testCKA(self):
        np_result = cka.cka(self.X, self.Y, kernel=cka.linear_kernel)
        tf_result = cka.cka_tf(self.X, self.Y, kernel=cka.linear_kernel_tf)

        self.assertAllClose(np_result, tf_result.numpy())


if __name__ == '__main__':
    tf.test.main()

