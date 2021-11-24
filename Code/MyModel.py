"""
@Date: 2021/11/23 下午5:51
@Author: Chen Zhang
@Brief: 自定义训练过程
"""
import tensorflow as tf


class MyModel(tf.keras.Model):

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # Compute the loss value
            # The loss function is configured in 'compile()'
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        output = {m.name: m.result() for m in self.metrics_names[:-1]}
        if 'iou' in self.metrics_names:
            self.metrics_names[-1].fill_output(output)
        return output

    def test_step(self, data):
        x, y = data

        y_pred = self(x, trainable=False)

        loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        output = {m.name: m.result() for m in self.metrics[:-1]}
        if 'iou' in self.metrics_names:
            self.metrics[-1].fill_output(output)
        return output

if __name__ == '__main__':
    pass
