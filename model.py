from keras.layers import *
import keras
import tensorflow as tf
import yaml
from xgboost import XGBClassifier
import lightgbm as lgb
from easydict import EasyDict as edict
import numpy as np
from easydict import EasyDict as edict

# Parameters
with open("config.yaml", "r") as f:
    cfg = edict(yaml.safe_load(f))

# gender, age, wt, ht
demo_shape = (cfg.train_param.demo_shape,)
# spo2, etco2, fio2, tv, pip, mv
input_shape = (cfg.train_param.input_shape[0], cfg.train_param.input_shape[1])
class_weight = 61223/5209787
#38848 / 3837411
#61223/5209787
INITIAL_BIAS = np.log(class_weight)


def build_gbm():
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric=["auc", "logloss"],
        tree_method="gpu_hist",
        use_label_encoder=False,
        scale_pos_weight=class_weight,
        seed=0,
        n_estimators=cfg.gbm.n_estimators,
        learning_rate=cfg.gbm.learning_rate,
        max_depth=cfg.gbm.max_depth,
        min_child_weight=cfg.gbm.min_child_weight,
        gamma=cfg.gbm.gamma,
        subsample=cfg.gbm.subsample,
        colsample_bytree=cfg.gbm.colsample_bytree,
    )



def build_transformer():
    # Input
    input = tf.keras.layers.Input(shape=input_shape)
    demo = tf.keras.layers.Input(shape=demo_shape)
    inputs = [input, demo]
    output = input

    # conv
    for _ in range(cfg.transformer.clayer):
        output = tf.keras.layers.Conv1D(
            filters=cfg.transformer.nfilt,
            kernel_size=cfg.transformer.filtsize,
            padding="same",
            activation="relu",
        )(output)
        output = tf.keras.layers.MaxPooling1D(cfg.transformer.poolsize, padding="same")(output)

    output = tf.keras.layers.Dense(cfg.transformer.kdim)(output)

    # transformer
    for _ in range(cfg.transformer.tlayer):
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=cfg.transformer.nhead,
            key_dim=cfg.transformer.kdim,
            attention_axes=[
                1,
            ],
        )(output, output)
        attn_output = tf.keras.layers.Dropout(cfg.transformer.droprate)(attn_output)
        # sum and norm
        output1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(output + attn_output)

        ffn_output = tf.keras.layers.Dense(cfg.transformer.fnode, activation="relu")(output1)
        ffn_output = tf.keras.layers.Dense(cfg.transformer.kdim)(ffn_output)
        output2 = tf.keras.layers.Dropout(cfg.transformer.droprate)(ffn_output)
        # sum and norm
        output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(output1 + output2)
    if cfg.transformer.pooltype == "avg":
        output = tf.keras.layers.GlobalAveragePooling1D()(output)
    else:
        output = tf.keras.layers.GlobalMaxPooling1D()(output)
    output = tf.keras.layers.Dropout(cfg.transformer.droprate)(output)

    output = tf.keras.layers.concatenate([demo, output])
    output = tf.keras.layers.Dense(cfg.transformer.fnode, activation="relu")(output)
    output = tf.keras.layers.Dropout(cfg.transformer.droprate)(output)
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(INITIAL_BIAS),
    )(output)

    return tf.keras.models.Model(inputs=[input, demo], outputs=[output])
    #return tf.keras.models.Model(inputs=input, outputs=[output])


class Classifier_INCEPTION:

    def __init__(self):
        self.nb_filters = cfg.inception.nb_filters
        self.use_residual = cfg.inception.use_residual
        self.use_bottleneck = cfg.inception.use_bottleneck
        self.depth = cfg.inception.depth
        self.kernel_size = cfg.inception.kernel_size - 1
        self.bottleneck_size = cfg.inception.bottleneck_size

    def _inception_module(self, input_tensor, stride=1, activation="linear"):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(
                filters=self.bottleneck_size,
                kernel_size=1,
                padding="same",
                activation=activation,
                use_bias=False,
            )(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2**i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(
                keras.layers.Conv1D(
                    filters=self.nb_filters,
                    kernel_size=kernel_size_s[i],
                    strides=stride,
                    padding="same",
                    activation=activation,
                    use_bias=False,
                )(input_inception)
            )

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding="same")(input_tensor)

        conv_6 = keras.layers.Conv1D(
            filters=self.nb_filters,
            kernel_size=1,
            padding="same",
            activation=activation,
            use_bias=False,
        )(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation="relu")(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(
            filters=int(out_tensor.shape[-1]),
            kernel_size=1,
            padding="same",
            use_bias=False,
        )(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation("relu")(x)
        return x

    def build_model(self):
        # Input
        input = tf.keras.layers.Input(shape=input_shape)
        demo = tf.keras.layers.Input(shape=demo_shape)
        inputs = [input, demo]
        output = input

        x = input
        input_res = input

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        output = keras.layers.GlobalAveragePooling1D()(x)
        output = tf.keras.layers.concatenate([demo, output])
        output = keras.layers.Dense(1, activation="sigmoid", bias_initializer=tf.keras.initializers.Constant(INITIAL_BIAS))(output)

        #model = keras.models.Model(inputs=input, outputs=output)
        model = keras.models.Model(inputs=inputs, outputs=output)

        return model


def build_lstm():
    # Input
    input = tf.keras.layers.Input(shape=input_shape)
    demo = tf.keras.layers.Input(shape=demo_shape)
    inputs = [input, demo]
    output = input

    # LSTM
    for _ in range(cfg.lstm.llayer - 1):
        output = tf.keras.layers.LSTM(cfg.lstm.lnode, return_sequences=True)(output)
    output = tf.keras.layers.LSTM(cfg.lstm.lnode)(output)

    # Linear
    output = tf.keras.layers.concatenate([demo, output])
    output = tf.keras.layers.Dense(cfg.lstm.fnode, activation="relu")(output)
    output = tf.keras.layers.Dropout(cfg.lstm.droprate)(output)
    output = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        bias_initializer=tf.keras.initializers.Constant(INITIAL_BIAS),
    )(output)

    #return tf.keras.models.Model(inputs=input, outputs=[output])
    return tf.keras.models.Model(inputs=inputs, outputs=[output])
