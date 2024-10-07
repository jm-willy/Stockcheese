import tensorflow as tf
import numpy as np
from datetime import datetime

# from tensorflow.keras.losses import mean_squared_error

optimizer = tf.keras.optimizers.Nadam(learning_rate=0.05)

from full_model import model


n_epochs = 1
batch_size = 2
batch_size = 2
n_batches = 2


# Custom training loop with NaN monitoring
@tf.function
def train_step(x_batch, y_batch, step):
    with tf.GradientTape() as tape:
        predictions = model(x_batch)
        # loss = tf.keras.losses.MeanSquaredError(y_batch, predictions)
        # loss = tf.keras.losses.mean_squared_error(y_batch, predictions)
        loss = tf.keras.losses.MSE(y_batch, predictions)

    # Get gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    # Check for NaN values in gradients
    has_nan = tf.reduce_any([tf.reduce_any(tf.math.is_nan(g)) for g in gradients])

    # If no NaNs, apply gradients
    def apply_gradients():
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # If NaNs detected, return high loss value
    def handle_nan():
        tf.print("\nNaN detected in gradients!")
        return tf.constant(1e10, dtype=tf.float32)

    loss = tf.cond(has_nan, handle_nan, apply_gradients)

    # Log metrics to TensorBoard
    with tf.summary_writer.as_default():
        tf.summary.scalar("loss", loss, step=step)
        for var, grad in zip(model.trainable_variables, gradients):
            tf.summary.histogram(f"{var.name}/gradients", grad, step=step)
            tf.summary.histogram(f"{var.name}/values", var, step=step)

    return loss


# Training loop
step = 0
X = np.random.rand(n_batches, 8, 8, 8, 1)
Y = np.random.rand(n_batches, 8, 8, 8, 1)
for epoch in range(n_epochs):
    print(f"\nEpoch {epoch + 1}/{n_epochs}")

    # Shuffle the data
    # indices = tf.random.shuffle(tf.range(len(X)))
    # X_shuffled = tf.gather(X, indices)
    # y_shuffled = tf.gather(Y, indices)

    for batch in range(n_batches):
        start_idx = batch * batch_size
        end_idx = start_idx + batch_size

        # x_batch = X_shuffled[start_idx:end_idx]
        # y_batch = y_shuffled[start_idx:end_idx]

        x_batch = X[start_idx:end_idx]
        y_batch = Y[start_idx:end_idx]

        loss = train_step(x_batch, y_batch, step)

        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")

        # If we detect a NaN (loss > 1e9), stop training
        if loss > 1e9:
            print("\nStopping training due to NaN detection")
            break

        step += 1

# print("\nTraining complete. To view the TensorBoard logs, run:")
# print(f"tensorboard --logdir {log_dir}")
