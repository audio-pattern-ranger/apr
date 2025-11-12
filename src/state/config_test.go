package state

import (
	"os"
	"testing"
)

func TestLoad_Configuration_WithTempFile(t *testing.T) {
	// Sample demo JSON configuration
	demoJSON := `{
		"workspace": "/tmp/demo_workspace",
		"audio_device": "demo_mic",
		"audio_options": ["-f", "alsa"],
		"video_device": "/dev/video9",
		"video_options": ["-f", "v4l2"],
		"video_advanced": ["-filter_complex", "[0:v]hflip"],
		"inspect_models": ["model1", "model2"],
		"inspect_backlog": 3,
		"inspect_segment": 10,
		"record_duration": "00:05:00",
		"train_rate": 0.005
	}`

	// Create a temporary file
	tmpFile, err := os.CreateTemp("", "config_*.json")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpFile.Name()) // delete after test

	// Write demo JSON to temp file
	if _, err := tmpFile.Write([]byte(demoJSON)); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	if err := tmpFile.Close(); err != nil {
		t.Fatalf("Failed to close temp file: %v", err)
	}

	// Load configuration from the temp file
	Load_Configuration(tmpFile.Name())

	cfg := Runtime

	// Verify some fields
	if cfg.Workspace != "/tmp/demo_workspace" {
		t.Errorf("Expected workspace '/tmp/demo_workspace', got '%s'", cfg.Workspace)
	}
	if cfg.Record_Audio_Device != "demo_mic" {
		t.Errorf("Expected Record_Audio_Device 'demo_mic', got '%s'", cfg.Record_Audio_Device)
	}
	if len(cfg.Record_Inspect_Models) != 2 {
		t.Errorf("Expected 2 inspect models, got %d", len(cfg.Record_Inspect_Models))
	}
	if cfg.Train_Rate != 0.005 {
		t.Errorf("Expected Train_Rate 0.005, got %f", cfg.Train_Rate)
	}
}
