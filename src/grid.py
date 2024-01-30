TRAINING_GRID = {
    "ver": (8,),
    "initialization": (
        "normal",
        "uniform",
    ),
    "hidden_size": (3,),
    "corpus_seed": (100,),
    "training_seed": (
        100,
        200,
        300,
        400,
        500,
    ),
    "regularization": (
        None,
        "L1",
        "L2",
    ),
    "regularization_lambda": (
        1.0,
        0.5,
        0.1,
    ),
    "learning_rate": (0.001,),
    "num_epochs": (20_000,),
    "dropout": (
        0.0,
        0.2,
        0.4,
        0.6,
    ),
    "validation_ratio": (0.05,),
    "early_stop_patience": (
        None,
        2,
        10,
    ),
    "train_size": (
        500,
        1000,
        5000,
        10000,
    ),
}
