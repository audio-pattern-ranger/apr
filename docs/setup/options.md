Configuration Options
=====================

Visit the [Configuration File](configure.md) section for information about
how to use these configuration options.

Workspace
---------

> This is the location where DTrack stores data. This includes saved recordings,
> audio clips, trained models, temporary data, etc.
>
> !!! option "Default Value: `"_workspace/"`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | string  | workspace              | DTRACK\_WORKSPACE         |


Workspace Keep Temp
-------------------

> Keep temporary data after extracted for review.
>
> !!! option "Default Value: `false`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | boolean | keep\_temp             | DTRACK\_KEEP\_TEMP        |

Record Audio Device
-------------------

> The device identifier used to record audio.
>
> !!! option "Default Value: `"plughw"`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | string  | audio\_device          | RECORD\_AUDIO\_DEVICE     |
>
> - **List audio devices:** `ffmpeg -loglevel warning -sources alsa`
>     + Sample Output:
>
>             Auto-detected sources for alsa:
>             null [Discard all samples (playback) or generate zero samples (capture)]
>             hw:CARD=Snowflake,DEV=0 [Direct hardware device without any conversions]
>             plughw:CARD=Snowflake,DEV=0 [Hardware device with all software conversions]
>             default:CARD=Snowflake [Default Audio Device]
>             sysdefault:CARD=Snowflake [Default Audio Device]
>             front:CARD=Snowflake,DEV=0 [Front output / input]
>             dsnoop:CARD=Snowflake,DEV=0 [Direct sample snooping device]
>
>     + Look for devices with: "<u>software conversions</u>"
>     + It is likely worth taking time to test available options.

Record Audio Options
--------------------

> Additional ffmpeg options used for audio capture.
>
> !!! option "Default Value: `[ "-f", "alsa" ]`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | list    | audio\_options         | RECORD\_AUDIO\_OPTIONS    |

Record Video Device
-------------------

> The device identifier used to record video.
>
> !!! option "Default Value: `"/dev/video0"`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | string  | video\_device          | RECORD\_VIDEO\_DEVICE     |
>
> - **List video devices:** `ls -1 /dev/video*`
>     + Sample Output:
>
>             /dev/video0
>             /dev/video1
>             /dev/video2
>             /dev/video3
>
> - **List capabilities:** `ffmpeg -hide_banner -f v4l2 -list_formats all -i <DEVICE>`
>     + Sample Output:
>
>             [v4l2 @ 0x5612] Compressed:       mjpeg: Motion-JPEG : 640x360 640x480 960x540 1024x576 1280x720 1920x1080 2560x1440 3840x2160
>             [v4l2 @ 0x5612] Compressed:        h264:       H.264 : 640x360 640x480 960x540 1024x576 1280x720 1920x1080 2560x1440 3840x2160
>             [v4l2 @ 0x5612] Compressed: Unsupported:        HEVC : 640x360 640x480 960x540 1024x576 1280x720 1920x1080 2560x1440 3840x2160
>             [v4l2 @ 0x5612] Raw       :     yuyv422:  YUYV 4:2:2 : 640x360 640x480 960x540 1024x576 1280x720 1920x1080
>             [in#0 @ 0x5612] Error opening input: Immediate exit requested
>             Error opening input file /dev/video2.
>
>     + This output shows a camera capable of recording 4K video.

Record Video Options
--------------------

> Additional ffmpeg options used for video capture.
>
> List video devices using: `ls /dev/video*`
>
> List video device capabilities using: `v4l2-ctl -d $DEVICE --list-formats-ext`;
> look for `'H264' (H.264, compressed)`.
>
> !!! option "Default Value: `[ "-f", "v4l2", "-input_format", "h264", "-video_size", "1920x1080", "-framerate", "20" ]`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | list    | video\_options         | RECORD\_VIDEO\_OPTIONS    |

Record Video Timestamp
----------------------

> Long string that defines the datestamp that is added on top of recorded video.
>
> The default value uses `Monospace Bold`, set to a bright `Red, 48-Point` font.
> This font is provided by the `fonts-freefont-ttf` package,
>
> This filter is applied to every frame, so it is possible to add `hflip,vflip,`
> to the start of the string, which will result in an image that is flipped both
> horizontally and vertically--perfect for an upside-down camera.
>
> !!! option "Default Value: `"drawtext=fontfile=/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf:text=%{localtime}:fontcolor=red@0.9:x=7:y=7:fontsize=48"`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | string  | video\_timestamp       | RECORD\_VIDEO\_TIMESTAMP  |

Record Video Advanced
---------------------

> Video creation arguments that ultimately control the final recorded video quality.
>
> A sensible range for 1080p video is between `3M` and `8M`, with `3.5M` being the
> observed upper limit on a Raspberry Pi 5--due to lack of hardware **en**coder.
>
> !!! option "Default Value: `[ "libx264", "-crf", "23", "-preset", "fast", "-tune", "zerolatency", "-maxrate", "3M", "-bufsize", "24M" ]`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | list    | video\_advanced        | RECORD\_VIDEO\_ADVANCED   |

Record Inspect Models
---------------------

> List of tags used for training, and trained models used for automatic detection.
>
> !!! option "Default Value: `[ ]`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | list    | inspect\_models        | RECORD\_INSPECT\_MODELS   |

Record Inspect Backlog
----------------------

> Defines the maximum number of "segments" that can back up in each queue
> before newly recorded audio is discarded from queue.
>
> !!! option "Default Value: `5`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | integer | inspect\_backlog       | RECORD\_INSPECT\_BACKLOG  |

Record Inspect Trust
----------------------

> Confidence level required to assume a match was found.
>
> !!! option "Default Value: `0.50`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | decimal | inspect\_trust         | RECORD\_INSPECT\_TRUST    |

Record Duration
---------------

> Length of time to use for each recording.
>
> - Very short recordings will suffer inadequate compression and are harder to review.
> - Very long recordings will be much larger and more difficult to submit.
>
> !!! option "Default Value: `"00:10:00"`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | string  | record\_duration       | RECORD\_DURATION          |

Train Batch Size
----------------

> Number of samples to use in each round of training.
>
> !!! option "Default Value: `16`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | integer | train\_batch\_size     | TRAIN\_BATCH\_SIZE        |

Train Epochs
------------

> Maximum number of learning attempts before stopping.
>
> !!! option "Default Value: `200`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | integer | train\_epochs          | TRAIN\_EPOCHS             |

Train Patience
--------------

> Maximum number of attempts to build a better model before stopping early.
>
> !!! option "Default Value: `10`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | integer | train\_patience        | TRAIN\_PATIENCE           |

Train Rate
----------

> Step size used to adjust learning. This may be decreased during training
> to prevent over-fitting data samples.
>
> !!! option "Default Value: `0.0001`"
>     | Type    | Configuration Variable | Environment Variable      |
>     | ------- | ---------------------- | ------------------------- |
>     | decimal | train\_rate            | TRAIN\_RATE               |
