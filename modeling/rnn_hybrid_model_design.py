from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HybridConfig:
    lookback_days: int = 30
    dynamic_feature_count: int = 5
    embedding_input_dim: int = 10
    embedding_mlp_hidden_dim: int = 32
    embedding_project_dim: int = 16
    rnn_units_1: int = 64
    rnn_units_2: int = 32
    dense_units: int = 64
    dropout: float = 0.2
    horizon: int = 7


MODEL_DESCRIPTION = """
Hybrid neural network design:
1) Dynamic branch (RNN input):
    - Input shape: (batch, lookback_days + 1, dynamic_feature_count)

2) Embedding branch (MLP projection):
    - Input shape: (batch, embedding_input_dim)
    - Dense(embedding_mlp_hidden_dim, activation='relu')
    - Dense(embedding_project_dim, activation='relu')
    - RepeatVector(lookback_days + 1)

3) Sequence fusion before LSTM:
    - Concatenate([dynamic_sequence, repeated_embedding_projection], axis=-1)
   - LSTM(rnn_units_1, return_sequences=True)
   - LSTM(rnn_units_2)

4) Final head:
   - Dense(dense_units, activation='relu')
   - Dropout(dropout)
   - Dense(horizon, activation='linear')

Both dynamic and embedding inputs are expected to be standardized upstream.
"""


def build_keras_model(config: HybridConfig):
    """Build the Keras model. Requires tensorflow to be installed in the environment."""
    import importlib

    tf = importlib.import_module("tensorflow")
    keras = importlib.import_module("tensorflow.keras")
    layers = keras.layers
    Model = keras.Model

    dynamic_input = layers.Input(
        shape=(config.lookback_days + 1, config.dynamic_feature_count),
        name="x_dynamic_seq",
    )

    embedding_input = layers.Input(shape=(config.embedding_input_dim,), name="x_embedding_std")
    x_emb = layers.Dense(config.embedding_mlp_hidden_dim, activation="relu", name="embedding_mlp_hidden")(embedding_input)
    x_emb = layers.Dense(config.embedding_project_dim, activation="relu", name="embedding_projection")(x_emb)
    x_emb = layers.RepeatVector(config.lookback_days + 1, name="repeat_embedding_over_time")(x_emb)

    x_seq = layers.Concatenate(axis=-1, name="concat_dynamic_and_embedding_seq")([dynamic_input, x_emb])
    x_seq = layers.LSTM(config.rnn_units_1, return_sequences=True, name="lstm_1")(x_seq)
    x_seq = layers.LSTM(config.rnn_units_2, name="lstm_2")(x_seq)

    fused = layers.Dense(config.dense_units, activation="relu", name="fusion_dense")(x_seq)
    fused = layers.Dropout(config.dropout, name="fusion_dropout")(fused)

    output = layers.Dense(config.horizon, activation="linear", name="forecast_horizon")(fused)

    model = Model(inputs=[dynamic_input, embedding_input], outputs=output, name="hybrid_rnn_embedding_model")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model


if __name__ == "__main__":
    cfg = HybridConfig()
    print(MODEL_DESCRIPTION)
    try:
        model = build_keras_model(cfg)
        model.summary()
    except Exception as exc:
        print("TensorFlow not available in this environment.")
        print(f"Build skipped: {exc}")
