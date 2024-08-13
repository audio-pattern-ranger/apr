Audio Pattern Ranger
====================

Audio Pattern Ranger (APR) offers 24/7 monitoring for local disturbances
in an environment, using machine learning models to detect and log specific
nuisances, such as barking or car alarms. These models are trained on
collected data to automate logging of detected disturbances.

Rather than using large complex solutions that make use of giant sample data,
APR uses local recordings in order to identify the exact disturbance. This means
that even an old laptop is plenty to put this project into action.

**Documentation**: https://audio-pattern-ranger.github.io/apr/

**Quickstart**::

    git clone https://github.com/audio-pattern-ranger/apr
    cd apr
    cp example_config.yml config.yml
    sensible-editor config.yml
    python3 -m apr --help

Background
----------

In some jurisdictions, understaffing can lead to a lack of support for
situations that are not life-threatening. In these cases, noise disturbances
may be entirely ignored without an extended log of repeated violation along
with video evidence proving log accuracy.

The primary purpose of this application is to simplify the collection and
analysis of video footage to identify disturbances (e.g., dog barks) using
a locally trained model. This model is designed to accurately detect and
classify specific disturbances in the local area.

How It Works
------------

1. Set up recorder
2. Collect some initial recordings
3. Extract individual noises
4. Train a model
5. Detect noises in collected recordings
6. Manually review the generated report
7. Refine model with additional training
