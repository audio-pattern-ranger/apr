package model_test

import (
	// DTrack
	"dtrack/model"

	// Standard
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Helper function to create a dummy raw audio buffer for dimensions testing
func createTestAudioBuffer(fillData bool) []byte {
	buffer := make([]byte, model.SampleSize)
	if fillData {
		for i := 0; i < model.SampleSize; i++ {
			buffer[i] = byte(i % 256)
		}
	}
	return buffer
}

// TestPrepare_Dimensions: Ensures DSP output has the correct shape [1, 3, 128, 188].
func TestPrepare_Dimensions(t *testing.T) {
	// Arrange: Create a valid input buffer (Silence)
	inputBuffer := createTestAudioBuffer(false)

	// Act: Call the Prepare function
	preparedTensor, err := model.Prepare(inputBuffer)
	if err != nil {
		t.Fatalf("Prepare failed during DSP processing: %v", err)
	}

	// Assert: Check the final expected shape
	expectedShape := []int{1, 3, model.Nmels, model.SpectrogramFrames}
	actualShape := preparedTensor.Shape()

	// Check dimensions count
	if len(actualShape) != len(expectedShape) {
		t.Fatalf("Dimension count mismatch. Expected %d, got %d", len(expectedShape), len(actualShape))
	}

	// Check specific dimensions (Channels, Height, Width)
	if actualShape[1] != expectedShape[1] || actualShape[2] != expectedShape[2] || actualShape[3] != expectedShape[3] {
		t.Errorf("Shape mismatch. Expected %v, got %v", expectedShape, actualShape)
	}

	t.Logf("Prepare() successful: Shape %v matches expected %v.", actualShape, expectedShape)
}

// TestMultiClass_Inference: Loads real files and verifies Multi-Class output.
func TestMultiClass_Inference(t *testing.T) {
	// Define paths for Real Artifacts
	onnxPath := "test_model.onnx"
	audioPath := "test_audio.dat"

	// Determine expected JSON path
	ext := filepath.Ext(onnxPath)
	jsonPath := strings.TrimSuffix(onnxPath, ext) + "_labels.json"

	// 1. Pre-Check: Ensure files exist to avoid log.Die() crash
	if _, err := os.Stat(onnxPath); os.IsNotExist(err) {
		t.Skipf("Skipping Test: Model file %s not found.", onnxPath)
	}
	if _, err := os.Stat(jsonPath); os.IsNotExist(err) {
		t.Skipf("Skipping Test: Labels file %s not found.", jsonPath)
	}
	if _, err := os.Stat(audioPath); os.IsNotExist(err) {
		t.Skipf("Skipping Test: Audio file %s not found.", audioPath)
	}

	// 2. Load Model (This will also load the JSON labels)
	myModel := model.Load(onnxPath)

	if len(myModel.Labels) < 2 {
		t.Errorf("Labels not loaded correctly. Found: %v", myModel.Labels)
	}
	t.Logf("Model loaded with classes: %v", myModel.Labels)

	// 3. Read Audio File
	rawBytes, err := os.ReadFile(audioPath)
	if err != nil {
		t.Fatalf("Could not read audio file: %v", err)
	}

	// 4. Prepare Audio (DSP)
	preparedTensor, err := model.Prepare(rawBytes)
	if err != nil {
		t.Fatalf("Model Prepare failed (DSP): %v", err)
	}

	// 5. Infer (Returns Map[string]float64)
	results := model.Infer(myModel, preparedTensor)

	// 6. Assertions
	t.Logf("Inference Results: %v", results)

	// Check if map is empty
	if len(results) == 0 {
		t.Fatal("Inference returned empty result map.")
	}

	// Check if all expected labels are present
	for _, label := range myModel.Labels {
		if _, ok := results[label]; !ok {
			t.Errorf("Missing probability for class: %s", label)
		}
	}

	sum := 0.0
	for _, prob := range results {
		sum += prob
	}

	if sum < 0.99 || sum > 1.01 {
		t.Errorf("Softmax failure: Probabilities sum to %f, expected ~1.0", sum)
	}
}
