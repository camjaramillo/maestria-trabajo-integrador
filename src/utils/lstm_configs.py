LSTM_CONFIGS = [
    # 1. LSTM simple (baseline)
    {
        "name": "lstm_simple",
        "layers": [64],
        "dropout": 0.2,
        "batch_size": 32,
        "epochs": 50,
        "optimizer": "adam"
    },

    # 2. LSTM profunda (stacked)
    {
        "name": "lstm_deep",
        "layers": [128, 64],
        "dropout": 0.3,
        "batch_size": 64,
        "epochs": 60,
        "optimizer": "adam",
        "recurrent_dropout": 0.2
    },

    # 3. LSTM bidireccional
    {
        "name": "lstm_bidirectional",
        "layers": [64],
        "bidirectional": True,
        "dropout": 0.2,
        "batch_size": 32,
        "epochs": 50,
        "optimizer": "rmsprop"
    },

    # 4. Stacked BiLSTM
    {
        "name": "bilstm_stacked",
        "layers": [128, 64],
        "bidirectional": True,
        "dropout": 0.3,
        "batch_size": 32,
        "epochs": 50,
        "optimizer": "adam"
    },

    # 5. CNN + LSTM
    {
        "name": "cnn_lstm",
        "conv1d": {"filters": 64, "kernel_size": 3, "activation": "relu"},
        "lstm_layers": [64],
        "dropout": 0.3,
        "batch_size": 64,
        "epochs": 60,
        "optimizer": "adam"
    },

    # 6. Attention LSTM
    {
        "name": "attention_lstm",
        "layers": [128],
        "dropout": 0.3,
        "batch_size": 32,
        "epochs": 70,
        "optimizer": "adam",
        "attention": True
    }
]