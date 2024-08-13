.. _index:

.. toctree::
   :hidden:
   :includehidden:

   About APR <self>
   Hardware <intro/hardware>
   Installation <intro/install>
   Configuration <intro/configure>
   How to Use <intro/usage>
   Troubleshooting <intro/troubleshooting>

.. _apr:

About APR
=========

**Audio Pattern Ranger (APR)** offers 24/7 monitoring for local disturbances
in an environment, using machine learning models to detect and log specific
nuisances, such as barking or car alarms. These models are trained on
collected data to automate logging of detected disturbances.

Rather than using large complex solutions that make use of giant sample data,
APR uses local recordings in order to identify the exact disturbance. This means
that even an old laptop is plenty to put this project into action.

.. _why:

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

.. _how-it-works:

How It Works
------------

  1. Set up recorder
  2. Collect some initial recordings
  3. Extract individual noises
  4. Train a model
  5. Detect noises in collected recordings
  6. Manually review the generated report
  7. Refine model with additional training

.. _getting-started:

Getting Started
---------------

  0. :ref:`Hardware <hardware>`
  1. :ref:`Install APR <installation>`
  2. :ref:`Configure <configuration>`
  3. :ref:`Usage (Tutorial) <usage>`
