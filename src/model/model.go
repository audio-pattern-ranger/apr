package model

import (
	// DTrack
	"dtrack/log"
	"dtrack/ffmpeg"

	// Standard
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"

	// 3rd-Party
	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/window"
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
)

const (
	// Number of seconds in each scanned segment
	SegmentSize       = 2

	// Full size of "check window"
	SampleSize        = ffmpeg.BytesPerSecond * SegmentSize

	// Spectrogram Values
	Nmels             = 128
	Nfft              = 2048
	HopLength         = 512
	SpectrogramFrames = 188
	Int16Max          = 32768.0

	// Frequency Limits
	MaxFreq = float64(ffmpeg.SampleRate) / 2
	MinFreq = 0.0
)

// OnnxModel holds raw bytes AND the class labels.
type OnnxModel struct {
	RawBytes []byte
	Labels   []string
}

// Not supported by golang
func Train() {
	log.Info("Model Training is handled by python -m ai.train")
}

// Load initializes the model bytes and loads the labels.json file.
func Load(model_path string) OnnxModel {
	log.Debug("Loading model from %s", model_path)

	// 1. Read .onnx file
	bytes, err := os.ReadFile(model_path)
	if err != nil {
		log.Die("could not read ONNX file: %s", err)
	}

	// 2. Read _labels.json file
	// e.g. whistle.onnx -> whistle_labels.json
	ext := filepath.Ext(model_path)
	jsonPath := strings.TrimSuffix(model_path, ext) + "_labels.json"

	labelsBytes, err := os.ReadFile(jsonPath)
	if err != nil {
		log.Die("could not read Labels file: %s", err)
	}

	var labels []string
	if err := json.Unmarshal(labelsBytes, &labels); err != nil {
		log.Die("could not parse Labels JSON: %s", err)
	}

	log.Debug("Loaded %s with classes: %v", model_path, labels)

	return OnnxModel{
		RawBytes: bytes,
		Labels:   labels,
	}
}

// Prepare takes raw audio bytes and converts them to a ready-to-infer tensor (DSP logic).
func Prepare(pcmData []byte) (*tensor.Dense, error) {
	// 1. Pad/Truncate the data to ensure fixed length (SampleSize)
	if len(pcmData) < SampleSize {
		log.Warn("Audio Underflow; Segment was not large enough!")
		padding := make([]byte, SampleSize-len(pcmData))
		pcmData = append(pcmData, padding...)
	}
	if len(pcmData) > SampleSize {
		log.Warn("Audio Overflow; Segment was too large!")
		pcmData = pcmData[:SampleSize]
	}

	// 2. Normalize to float64 for DSP
	audioFloat32 := normalizeAudio(pcmData)
	audioFloat64 := make([]float64, len(audioFloat32))
	for i, v := range audioFloat32 {
		audioFloat64[i] = float64(v)
	}

	// 3. STFT Calculation
	n_frames_calculated := (len(audioFloat64)-Nfft)/HopLength + 1

	// Check for extreme case where calculated frames exceed expected
	if n_frames_calculated > SpectrogramFrames {
		log.Warn("ML: Calculated frames exceed expected. Using calculated size.")
		n_frames_calculated = SpectrogramFrames
	}

	// Calculate bins to keep based on Full Spectrum (Matches Python)
	bins_linear := Nfft/2 + 1 // 1025 bins

	// Create Power Spectrogram (bins_linear * frames)
	powerSpectrogram := make([][]float64, bins_linear)
	for i := range powerSpectrogram {
		powerSpectrogram[i] = make([]float64, n_frames_calculated)
	}

	window_func := window.Hann(Nfft)

	for i := 0; i < n_frames_calculated; i++ {
		start := i * HopLength
		end := start + Nfft
		if end > len(audioFloat64) {
			end = len(audioFloat64)
		}

		segment := audioFloat64[start:end]
		windowed := make([]float64, Nfft)
		for j := 0; j < len(segment); j++ {
			windowed[j] = segment[j] * window_func[j]
		}

		complex_input := make([]complex128, Nfft)
		for j := 0; j < Nfft; j++ {
			complex_input[j] = complex(windowed[j], 0)
		}
		complex_result := fft.FFT(complex_input)

		// Magnitude Square
		for j := 0; j < bins_linear; j++ {
			val := complex_result[j]
			magSq := real(val)*real(val) + imag(val)*imag(val)
			powerSpectrogram[j][i] = magSq
		}
	}

	// 4. Convert to Mel Spectrogram
	melSpectrogram := applyMelFilterbank(powerSpectrogram, bins_linear, n_frames_calculated)
	melDb := powerToDb(melSpectrogram)

	// 5. Normalize for tensor (-80db floor)
	normalized := fixedNormalize(melDb)
	flatData := make([]float32, Nmels * SpectrogramFrames)

	idx := 0
	for r := 0; r < Nmels; r++ {
		for c := 0; c < SpectrogramFrames; c++ {
			if c < n_frames_calculated {
				flatData[idx] = float32(normalized[r][c])
			} else {
				//log.Warn("Padding used during data normalization")
				flatData[idx] = 0.0
			}
			idx++
		}
	}

	// Final tensor shape: [Batch, Channel, Height, Width]
	shape := []int{1, 1, Nmels, SpectrogramFrames}


	inputTensor := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(shape...),
		tensor.WithBacking(flatData),
	)

	return inputTensor, nil
}

// Infer runs the model and returns a MAP of probabilities (Multi-Class).
// Returns: map["barking"] = 0.8, map["empty"] = 0.2
func Infer(inferModel OnnxModel, preparedAudio *tensor.Dense) map[string]float64 {
	// 1. Create Backend
	backend := gorgonnx.NewGraph()
	onnxModel := onnx.NewModel(backend)

	// 2. Unmarshal
	if err := onnxModel.UnmarshalBinary(inferModel.RawBytes); err != nil {
		log.Die("could not unmarshal ONNX model: %s", err)
	}

	// 3. Run
	onnxModel.SetInput(0, tensor.Tensor(preparedAudio))
	if err := backend.Run(); err != nil {
		log.Die("Inference failed: %v", err)
	}

	// 4. Get Output
	outputTensors, _ := onnxModel.GetOutputTensors()
	outputDense, ok := outputTensors[0].(*tensor.Dense)
	if !ok {
		log.Die("Output tensor is not a *tensor.Dense type.")
	}

	// 5. Convert Logits to Probabilities (Softmax)
	floatSlice := outputDense.Data().([]float32) // Gorgonia usually returns float32
	logits := make([]float64, len(floatSlice))
	for i, v := range floatSlice {
		logits[i] = float64(v)
	}

	probs := softmax(logits)

	// 6. Map to Labels
	results := make(map[string]float64)
	for i, label := range inferModel.Labels {
		if i < len(probs) {
			results[label] = probs[i]
		}
	}

	return results
}

func softmax(logits []float64) []float64 {
	max := -math.MaxFloat64
	for _, v := range logits {
		if v > max {
			max = v
		}
	}
	sum := 0.0
	exps := make([]float64, len(logits))
	for i, v := range logits {
		exps[i] = math.Exp(v - max)
		sum += exps[i]
	}
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}

// Convert raw 16-bit PCM bytes into a normalized float32 slice
func normalizeAudio(pcmData []byte) []float32 {
	numSamples := len(pcmData) / 2
	audioArray := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		val := int16(uint16(pcmData[i*2]) | uint16(pcmData[i*2+1])<<8)
		audioArray[i] = float32(val) / Int16Max
	}
	return audioArray
}
