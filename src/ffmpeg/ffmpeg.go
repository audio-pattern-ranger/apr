package ffmpeg

import (
	// DTrack
	"dtrack/log"
	"dtrack/state"

	// Standard
	"bytes"
	"io"
	"os"
	"os/exec"
	"syscall"
	"time"
)

// One-second segment of audio from pcm_s16le (segment_size)
//
//	Bytes Per Second = Sample Rate * Channels * (Bits Per Sample / 8)
//	96000            = -ac 48000   * -c 1     * (16/8)
const BytesPerSecond int = 96000

// MKV Filename:  YYYY-MM-DD_HHmmss
const SaveName = "2006-01-02_150405.mkv"

// Run ffmpeg command, returning stdout to IO stream
func ReadStdin(arguments []string, stdout *io.PipeWriter, endStream bool) {
	if endStream {
		defer stdout.Close()
	}
	ffmpeg := exec.Command("ffmpeg", arguments...)
	ffmpeg.Stderr = os.Stderr
	ffmpeg.Stdout = stdout

	// Use separate process group to avoid SIGTERM collisions
	ffmpeg.SysProcAttr = &syscall.SysProcAttr{
		Setpgid: true,
	}

	// Start ffmpeg process
	if ffmpeg.Start() != nil {
		log.Die("Failed to intialize ffmpeg")
	}
	if ffmpeg.Wait() != nil {
		log.Warn("ffmpeg finished with errors")
		// Extra pause for potential device thrashing
		time.Sleep(1 * time.Second)
	}
}

// Run ffplay command, writing from IO stream
func PlayData(wavData []byte) {
	ffplay := exec.Command("aplay")
	//ffplay.Stderr = os.Stderr
	ffplay.Stdin = bytes.NewReader(wavData)

	// Run ffplay process
	if _, err := ffplay.CombinedOutput(); err != nil {
		log.Warn("Error playing audio clip: %s", err)
	}
}

// Return list of arguments for ffmpeg that:
//
//	Reads Audio to Stream and Video to Images.
//
//	ffmpeg [basic-options] [input-mkv] \
//	  [wav-to-stdout] \
//	  [output-wav] \
//	  [output-images]
func Extract_Arguments(infile string, outdir string) []string {
	return []string{
		// basic-options input-mkv
		"-y", "-loglevel", "warning", "-nostdin", "-nostats", "-i", infile,
		// wav-to-stdout
		"-map", "0:a:0", "-f", "s16le", "-ar", "48000", "-ac", "1", "-",
		// output-wav
		"-f", "segment", "-segment_time", "1", "-reset_timestamps", "1", outdir + "/%d.wav",
		// output-images
		"-map", "0:v:0", "-vf", "fps=1,scale=1536:864", "-start_number", "0", outdir + "/%d.jpg"}
}

// Return list of arguments for ffmpeg that:
//
//	Saves A/V to MKV and Audio to Stream.
//
//	ffmpeg [basic-options] \
//	  [audio-options] [audio-device] \
//	  [video-options] [video-device] \
//	  [output-wav] [to-stdout] \
//	  [output-wav&vid] [to-mkv] [MISSING:filename]
func Recorder_Arguments() []string {
	// basic-options
	args := []string{
		"-y", "-loglevel", "warning", "-nostdin", "-nostats",
		"-guess_layout_max", "1"}

	// audio-options
	args = append(args, "-t", state.Runtime.Record_Duration)
	args = append(args, state.Runtime.Record_Audio_Options...)
	// audio-device
	args = append(args, "-i", state.Runtime.Record_Audio_Device)

	// video-options
	args = append(args, "-t", state.Runtime.Record_Duration)
	args = append(args, state.Runtime.Record_Video_Options...)
	// video-device
	args = append(args, "-i", state.Runtime.Record_Video_Device)

	// wav-to-stdout
	if state.Runtime.Has_Models {
		args = append(args,
			"-map", "0:a", "-c:a", "pcm_s16le",
			"-ar", "48000", "-ac", "1", "-f", "wav", "-")
	}
	// wav&vid-to-mkv
	args = append(args,
		"-filter_complex", "[1:v]" + state.Runtime.Record_Video_Timestamp + "[dtstamp]",
		"-map", "0:a", "-map", "[dtstamp]", "-c:a", "pcm_s16le",
		"-ar", "48000", "-ac", "1", "-c:v")
	args = append(args, state.Runtime.Record_Video_Advanced...)

	log.Debug("Compiled recorder arguments: %s", args)
	return args
}
