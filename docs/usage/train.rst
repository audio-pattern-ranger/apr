.. _train:

Model Training
--------------

Training a model is essentially a continuous loop of making a random model and
then comparing it's effectiveness to the current best.

  .. note::

    If a model already exists, it will be used to prime the training routine.

After all testing and training data is generated, the process can be initiated
with:

  .. code-block:: sh

    python3 -m apr -a train

This will continue until ``target_accuracy`` (from ``config.yml``) is met.

  .. code-block:: text

    python3 -m apr -a train
    INFO:Training iteration 1
    INFO:Overall accuracy[1] is 50.0
    INFO:Accuracy increased; keeping new model
    INFO:Training iteration 2
    INFO:Overall accuracy[2] is 50.0
    INFO:Accuracy worse than #1; discarding new
    INFO:Training iteration 3
    [...]
    INFO:Training iteration 11
    INFO:Overall accuracy[11] is 84.84848484848484
    INFO:Accuracy worse than #9; discarding new
    INFO:Training iteration 12
    INFO:Overall accuracy[12] is 86.36363636363637
    INFO:Accuracy increased; keeping new model
    INFO:TRAINING COMPLETE :: Final Accuracy: 86.36363636363637

The final products of this training process are ``model.pth`` and ``model.wav``.
These two files can be copied into another workspace and then used for
:ref:`content inspection (detection) <inspect>`.
