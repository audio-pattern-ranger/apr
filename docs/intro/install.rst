.. _installation:

Installation
============

APR can be installed on any OS supporting Python 3 and PyTorch.

Debian Dependencies
-------------------

Needed by **all**:

  .. code-block:: sh

    sudo apt install ffmpeg v4l-utils \
        python3-yaml python3-pydub python3-torch python3-torchaudio

Needed to :ref:`**monitor** <recording>`:

  .. code-block:: sh

    sudo apt install python3-fasteners
    # Font is optional; used by record_cam_timestamp in config.yml
    sudo apt install fonts-freefont-ttf

Needed to :ref:`inspect and train <training>`:

  .. code-block:: sh

    sudo apt install python3-tk python3-ttkthemes python3-pil.imagetk python3-moviepy

Python VirtualEnv
-----------------

An alternative to installing OS packages is to install using ``pip``:

  .. code-block:: sh

    # Install virtualenv and build dependencies
    sudo apt install -y python3-virtualenv build-essenial \
        python3-dev libasound2-dev

    # Create an initial environment
    python3 -m venv ~/.mlpy

    # Load virtual environment
    source  ~/.mlpy/bin/activate

    # Build dependencies
    pip3 install -r requirements.txt

APR Source Code
---------------

The easiest way to obtain APR is using git:

  .. code-block:: sh

    git clone https://github.com/audio-pattern-ranger/apr

.. _install-verification:

Verification
------------

Successful installation can be verified by viewing help text:

  .. code-block:: sh

    (.mlpy) michael@vsense1:~/apr $ python3 -m apr --help
    usage: apr [-h] -a <action> [other_options]
    [...]
