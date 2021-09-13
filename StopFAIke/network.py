import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_addons as tfa


def build_classifier_model(tfhub_handle_encoder):
    class Classifier(tf.keras.Model):
        def __init__(self, tfhub_handle_encoder):
            super(Classifier, self).__init__(name="prediction")
            self.encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True)
            self.dropout = tf.keras.layers.Dropout(0.5)
            self.dense_int = tf.keras.layers.Dense(128, activation='relu')
            self.dense_final = tf.keras.layers.Dense(1, activation='sigmoid')

        def call(self, preprocessed_text):
            encoder_outputs = self.encoder(preprocessed_text)
            pooled_output = encoder_outputs["pooled_output"]
            x = self.dropout(pooled_output)
            x = self.dense_int(x)
            x = self.dropout(x)
            x = self.dense_final(x)
            return x

    model = Classifier(tfhub_handle_encoder)

    return model
