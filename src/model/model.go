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
	SegmentSize       = 2
	SampleSize        = ffmpeg.BytesPerSecond * SegmentSize
	Nmels             = 128
	Nfft              = 2048
	HopLength         = 512
	SpectrogramFrames = 188
	Int16Max          = 32768.0
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
	bins_to_keep := Nfft/2 + 1 // 1025 bins

	// Create Spectrogram Matrix
	spectrogram := make([][]float64, bins_to_keep)
	for i := range spectrogram {
		spectrogram[i] = make([]float64, SpectrogramFrames)
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
		for j := 0; j < bins_to_keep; j++ {
			val := complex_result[j]
			magSq := real(val)*real(val) + imag(val)*imag(val)
			spectrogram[j][i] = magSq
		}
	}

	// 4. Flatten & Resize
	power_spec_1D := make([]float64, bins_to_keep*SpectrogramFrames)
	for i := 0; i < bins_to_keep; i++ {
		copy(power_spec_1D[i*SpectrogramFrames:(i+1)*SpectrogramFrames], spectrogram[i])
	}

	// Resize using Average Pooling
	rescaled_spec := resizeRowsAvg(power_spec_1D, bins_to_keep, SpectrogramFrames, Nmels)

	// 5. Power to DB & Normalize
	mel_spec_db := powerToDb(rescaled_spec)
	img_normalized := minMaxNormalize(mel_spec_db)

	// 6. Stack Channels
	size := len(img_normalized)
	img3Channel := make([]float32, size*3)
	copy(img3Channel[:size], img_normalized)
	copy(img3Channel[size:size*2], img_normalized)
	copy(img3Channel[size*2:size*3], img_normalized)

	shape := []int{1, 3, Nmels, SpectrogramFrames}

	inputTensor := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(shape...),
		tensor.WithBacking(img3Channel),
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

// Convert a power spectrogram to decibels
func powerToDb(spec []float64) []float64 {
	dbSpec := make([]float64, len(spec))
	var maxPower float64 = 0.0
	for _, v := range spec {
		if v > maxPower {
			maxPower = v
		}
	}
	if maxPower < 1e-10 {
		maxPower = 1.0
	}
	for i, v := range spec {
		dbSpec[i] = 10.0 * math.Log10(math.Max(v/maxPower, 1e-10))
	}
	return dbSpec
}

func minMaxNormalize(dbSpec []float64) []float32 {
	var minVal float64 = math.MaxFloat64
	var maxVal float64 = -math.MaxFloat64
	for _, v := range dbSpec {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	rangeVal := maxVal - minVal
	if rangeVal < 1e-6 {
		rangeVal = 1e-6
	}
	img := make([]float32, len(dbSpec))
	for i, v := range dbSpec {
		img[i] = float32((v - minVal) / rangeVal)
	}
	return img
}

func resizeRowsAvg(input []float64, originalRows, cols, targetRows int) []float64 {
	if originalRows == targetRows {
		return input
	}
	output := make([]float64, targetRows*cols)
	ratio := float64(originalRows) / float64(targetRows)

	for c := 0; c < cols; c++ {
		for r_target := 0; r_target < targetRows; r_target++ {
			startRow := int(math.Floor(float64(r_target) * ratio))
			endRow := int(math.Ceil(float64(r_target+1) * ratio))
			if endRow > originalRows {
				endRow = originalRows
			}

			sum := 0.0
			count := 0
			for r_orig := startRow; r_orig < endRow; r_orig++ {
				sum += input[r_orig*cols+c]
				count++
			}
			if count > 0 {
				output[r_target*cols+c] = sum / float64(count)
			}
		}
	}
	return output
}
