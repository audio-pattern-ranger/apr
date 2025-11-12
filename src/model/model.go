package model

import (
	"github.com/owulveryck/onnx-go"
	"github.com/owulveryck/onnx-go/backend/x/gorgonnx"
	"gorgonia.org/tensor"
	"github.com/mjibson/go-dsp/fft"
	"github.com/mjibson/go-dsp/window"

	// DTrack
	"dtrack/log"

	// Standard
	"fmt"
	"math"
	"os"
)

const (
	SAMPLE_RATE        = 48000
	AUDIO_LENGTH_BYTES = 192000
	N_MELS             = 128
	N_FFT              = 2048
	HOP_LENGTH         = 512
	SPECTROGRAM_FRAMES = 188
	INT16_MAX          = 32768.0
)

type loaded_model struct {
	onnxModel *onnx.Model
	backend   *gorgonnx.Graph
}

// Not supported by golang
func Train() {
	log.Info("Model Training is handled by python -m ai.train")
}

// Loads the ONNX model once and initializes the Gorgonia backend.
func Load(model_path string) loaded_model {
	log.Debug("Loading model from %s", model_path)

	// Read model data from onnx file
	bytes, err := os.ReadFile(model_path)
	if err != nil {
		log.Die("could not read ONNX file: %s", err)
	}

	// Initialize Gorgonia backend
	gorgoniaBackend := gorgonnx.NewGraph()

	// Load model from onnx data
	onnxModel := onnx.NewModel(gorgoniaBackend)
	if err := onnxModel.UnmarshalBinary(bytes); err != nil {
		log.Die("could not unmarshal ONNX model: %s", err)
	}

	// Return the loaded model
	return loaded_model{
		onnxModel: onnxModel,
		backend:   gorgoniaBackend,
	}
}

// Prepare takes raw audio bytes and converts them to a ready-to-infer tensor (DSP logic).
func Prepare(pcmData []byte) (*tensor.Dense, error) {
	// 1. Pad/Truncate the data to ensure fixed length (AUDIO_LENGTH_BYTES)
	//if len(pcmData) < AUDIO_LENGTH_BYTES {
	//	padding := make([]byte, AUDIO_LENGTH_BYTES-len(pcmData))
	//	pcmData = append(pcmData, padding...)
	//}
	//if len(pcmData) > AUDIO_LENGTH_BYTES {
	//	pcmData = pcmData[:AUDIO_LENGTH_BYTES]
	//}

	// 2. Normalize to float64 for DSP
	audioFloat32 := normalizeAudio(pcmData)
	audioFloat64 := make([]float64, len(audioFloat32))
	for i, v := range audioFloat32 {
		audioFloat64[i] = float64(v)
	}

	// 3. STFT (Power Spectrogram)
	// Calculate the actual number of frames that can be calculated (184 in your case)
	n_frames_calculated := (len(audioFloat64)-N_FFT)/HOP_LENGTH + 1

	if n_frames_calculated <= 0 {
		return nil, fmt.Errorf("audio clip is too short for STFT")
	}

	n_frames := SPECTROGRAM_FRAMES

	// Check for extreme case where calculated frames exceed expected
	if n_frames_calculated > SPECTROGRAM_FRAMES {
		n_frames = n_frames_calculated
		log.Warn("ML: Calculated frames exceed expected. Using calculated size.")
	}

	rows_orig := N_FFT/2 + 1 // 1025 rows
	// Spectrogram array ki final size 1025 x 188 hogi (last 4 columns zero-filled rahenge)
	spectrogram := make([][]float64, rows_orig)
	for i := range spectrogram {
		spectrogram[i] = make([]float64, n_frames)
	}

	window_func := window.Hann(N_FFT)

	// Loop sirf calculate kiye gaye frames tak chalega (184 frames)
	for i := 0; i < n_frames_calculated; i++ {
		start := i * HOP_LENGTH
		end := start + N_FFT
		if end > len(audioFloat64) {
			end = len(audioFloat64)
		}
		segment := audioFloat64[start:end]

		windowed := make([]float64, N_FFT)
		for j := 0; j < len(segment); j++ {
			windowed[j] = segment[j] * window_func[j]
		}

		// FFT Conversion and Calculation
		complex_input := make([]complex128, N_FFT)
		for j := 0; j < N_FFT; j++ {
			complex_input[j] = complex(windowed[j], 0)
		}
		complex_result := fft.FFT(complex_input)

		// Magnitude Square (Power Spectrogram)
		for j := 0; j < rows_orig; j++ {
			spectrogram[j][i] = real(complex_result[j])*real(complex_result[j]) + imag(complex_result[j])*imag(complex_result[j])
		}
	}

	// 4. Rescale/Resize
	power_spec_1D := make([]float64, rows_orig*n_frames) // Final size 1025 * 188
	for i := 0; i < rows_orig; i++ {
		copy(power_spec_1D[i*n_frames:(i+1)*n_frames], spectrogram[i])
	}

	rescaled_spec := resizeRows(power_spec_1D, rows_orig, n_frames, N_MELS)
	rows := N_MELS

	// 5. Power to DB & Min-Max Normalize
	mel_spec_db := powerToDb(rescaled_spec, rows, n_frames)
	img_normalized := minMaxNormalize(mel_spec_db)

	// 6. Stack Channels & Convert to Tensor
	size := len(img_normalized)
	img3Channel := make([]float32, size*3)
	copy(img3Channel[:size], img_normalized)
	copy(img3Channel[size:size*2], img_normalized)
	copy(img3Channel[size*2:size*3], img_normalized)

	shape := []int{1, 3, N_MELS, n_frames}

	inputTensor := tensor.New(
		tensor.Of(tensor.Float32),
		tensor.WithShape(shape...),
		tensor.WithBacking(img3Channel),
	)

	return inputTensor, nil
}

// Infer runs the prepared audio tensor through the ONNX model.
func Infer(myModel loaded_model, preparedAudio *tensor.Dense) float64 {
	inputInterface := tensor.Tensor(preparedAudio)

	// 1. Set Input and Run Backend
	myModel.onnxModel.SetInput(0, inputInterface)
	if err := myModel.backend.Run(); err != nil {
		log.Die("ML: Inference failed on Gorgonia backend: %v", err)
	}

	// 2. Get Output and Post-Process (Sigmoid)
	outputTensors, _ := myModel.onnxModel.GetOutputTensors()
	outputDense, ok := outputTensors[0].(*tensor.Dense)
	if !ok {
		log.Die("ML: Output tensor is not a *tensor.Dense type.")
	}

	backingSlice := outputDense.Data()
	floatSlice, ok := backingSlice.([]float32)
	if !ok || len(floatSlice) == 0 {
		log.Die("ML: Output data corrupted or empty.")
	}

	logitFloat := float64(floatSlice[0])
	probability := 1.0 / (1.0 + math.Exp(-logitFloat))

	return probability
}

// normalizeAudio converts raw 16-bit PCM bytes into a normalized float32 slice.
func normalizeAudio(pcmData []byte) []float32 {
	numSamples := len(pcmData) / 2
	audioArray := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		val := int16(uint16(pcmData[i*2]) | uint16(pcmData[i*2+1])<<8)
		audioArray[i] = float32(val) / INT16_MAX
	}
	return audioArray
}

// powerToDb converts a power spectrogram to decibels.
func powerToDb(spec []float64, rows, cols int) []float64 {
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

// minMaxNormalize scales the decibel spectrogram to a 0-1 range.
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

// resizeRows: Resizes the spectrogram rows (using Average Pooling Hack)
func resizeRows(input []float64, originalRows, originalCols, targetRows int) []float64 {
	if originalRows == targetRows {
		return input
	}
	output := make([]float64, targetRows*originalCols)
	ratio := float64(originalRows) / float64(targetRows)

	for c := 0; c < originalCols; c++ {
		for r_target := 0; r_target < targetRows; r_target++ {
			startRow := int(math.Floor(float64(r_target) * ratio))
			endRow := int(math.Ceil(float64(r_target+1) * ratio))
			if endRow > originalRows {
				endRow = originalRows
			}

			sum := 0.0
			count := 0
			for r_orig := startRow; r_orig < endRow; r_orig++ {
				sum += input[r_orig*originalCols+c]
				count++
			}
			if count > 0 {
				output[r_target*originalCols+c] = sum / float64(count)
			}
		}
	}
	return output
}
