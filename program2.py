import tensorflow as tf


a = tf.constant(2)
b = tf.constant(3)


add = tf.add(a, b)
sub = tf.subtract(a, b)
mul = tf.multiply(a, b)
div = tf.divide(a, b)

print("Addition:", add.numpy())
print("Subtraction:", sub.numpy())
print("Multiplication:", mul.numpy())
print("Division:", div.numpy())


mat1 = tf.constant([[1, 2], [3, 4]])
mat2 = tf.constant([[5, 6], [7, 8]])


mat_add = tf.add(mat1, mat2)


mat_mul = tf.matmul(mat1, mat2)


mat_transpose = tf.transpose(mat1)


mat_reshape = tf.reshape(mat1, [4, 1])

print("Matrix Addition:\n", mat_add.numpy())
print("Matrix Multiplication:\n", mat_mul.numpy())
print("Transpose:\n", mat_transpose.numpy())
print("Reshape:\n", mat_reshape.numpy())

@tf.function
def my_operation(x, y):
    return tf.multiply(x, y) + tf.add(x, y)

result = my_operation(4, 5)
print("Result from custom function:", result.numpy())

tf.compat.v1.disable_eager_execution()  # Turn off eager execution

a = tf.compat.v1.constant(10)
b = tf.compat.v1.constant(20)
add_op = tf.add(a, b)


with tf.compat.v1.Session() as sess:
    result = sess.run(add_op)
    print("Session Result:", result)
