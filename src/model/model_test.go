package model_test

import (
	// DTrack
	"dtrack/model"

	// Standard
	"math"
	"testing"
	"os"
)

// Helper function to create a dummy raw audio buffer for testing
func createTestAudioBuffer(fillData bool) []byte {
	buffer := make([]byte, model.AUDIO_LENGTH_BYTES)

	if fillData {
		for i := 0; i < model.AUDIO_LENGTH_BYTES; i++ {
			buffer[i] = byte(i % 256)
		}
	} else {
		// Zero-filled buffer (Silence)
		for i := 0; i < model.AUDIO_LENGTH_BYTES; i++ {
			buffer[i] = 0
		}
	}

	return buffer
}

// --- UNIT TESTS ---

// TestPrepare_Dimensions: Ensures DSP output has the correct shape.
func TestPrepare_Dimensions(t *testing.T) {
	// Arrange: Create a valid input buffer (Silence)
	inputBuffer := createTestAudioBuffer(false)

	// Act: Call the Prepare function
	preparedTensor, err := model.Prepare(inputBuffer)

	if err != nil {
		t.Fatalf("Prepare failed during DSP processing: %v", err)
	}

	// Assert: Check the final expected shape
	expectedShape := []int{1, 3, model.N_MELS, model.SPECTROGRAM_FRAMES}
	actualShape := preparedTensor.Shape()

	if actualShape[len(actualShape)-1] != expectedShape[len(expectedShape)-1] {
		// This log helps debug the 184 vs 188 frame mismatch
		t.Errorf("FATAL FRAME MISMATCH. Expected frame count %d, got %d. (Fix needed in model.Prepare logic)", expectedShape[len(expectedShape)-1], actualShape[len(actualShape)-1])
	}

	if len(actualShape) != len(expectedShape) || actualShape[2] != expectedShape[2] {
		t.Fatalf("Shape mismatch. Expected %v, got %v.", expectedShape, actualShape)
	}

	t.Logf("Prepare() successful: Shape %v matches expected %v.", actualShape, expectedShape)
}

// TestFullInference_KnownOutput: Tests the full pipeline (Init + Prepare + Infer) against a known output range.
func TestFullInference_KnownOutput(t *testing.T) {
	// Arrange: Use the testing file's raw bytes
	rawBytes, err := os.ReadFile("test_audio.dat")
	if err != nil {
		t.Fatalf("Skipping test: Could not read test file: %v", err)
	}

	// 1. Initialize Model (Must succeed)
	myModel := model.Load("test_model.onnx")

	// 2. Prepare Audio (DSP)
	preparedTensor, err := model.Prepare(rawBytes)
	if err != nil {
		t.Fatalf("Model Prepare failed (DSP): %v", err)
	}

	// 3. Infer (Full Inference Run)
	confidence := model.Infer(myModel, preparedTensor)

	// 4. Assert: Check the output range and expected prediction (based on your Python run)

	// Python Result for this file: Hit Probability: 0.998199
	// Go Result (Observed): Hit Probability: 0.9972 - 0.999

	// We check if the confidence is high and within a reasonable range (>= 90%)
	if confidence < 0.90 || confidence > 1.0 {
		t.Fatalf("Inference failed: Confidence is outside expected range. Expected > 0.90, Got %.4f", confidence)
	}

	t.Logf("Full Inference Success! Final Confidence: %.4f (Expected high match)", confidence)
}

// sigmoid function from model.go (needed for local validation)
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
