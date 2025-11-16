Hardware Selection
==================

The financial investment required for this project comes down to hardware
selection.

The hardware required:

- Training Device: Typically the same laptop being used to set things up
- Monitoring Device: Low-power mini-PC with at least two (2) USB ports:
    * Microphone: The sensor that detects disturbing audio
    * Camera: The sensor that shows proof of audio source

Training Device
---------------

A laptop (or desktop) of any age will work great for creating a trained model.
CUDA-supporting GPUs, like Nvidia, can improve training time by up to 20%.

Any currently-owned laptop with a webcam and microphone is likely a great option
to try out this project, before deciding to purchase any additional hardware.

Monitoring Device
-----------------

Although much of DTrack can easily run on a Raspberry Pi, these devices lack a
"hardware video **en**coder", making video generation a very intensive process.
This limits recording to about 3M/s, which is the extreme low side of 1080p.

Fortunately, there are countless mini-PCs with hardware **en**coders that can
tackle high-quality 4k video with ease, making storage size the next bottleneck.

Any mini-PC with one of the following CPUs should be great picks:

- Intel Celeron (N150+)
- Intel Gemini Lake (N4000+)
- Intel Jasper Lake (N5000+)
- AMD Ryzen Embedded (R2000+, 7000+, 8000+)

**Microphone:**

The trained model used for automatic detection will be created using audio samples
that were created from this microphone. This makes it one of the most important
hardware decisions for this project. Changing this will require recording new
samples in order to train a new model.

Many microphones are designed to eliminate audio that is likely to be clasified
as disturbances, such as dog barks or car alarms. It is important to find a
microphone that does not advertise features like noise cancelling and instead
offers "high sensitivity" and a "wide frequency response."

Some great options include:

- Angetube USB Microphone
- Blue Snowflake

**Camera:**

The camera you choose must be able to capture enough detail to identify the origin
of the noise, but that is all the detail that is required. There is no reason to
find an expensive camera.

Most modern webcams that claim 1080p will be capable of capturing enough detail
to accompany a written report.
