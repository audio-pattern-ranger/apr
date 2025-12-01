package model

import (
	// DTrack
	"dtrack/ffmpeg"

	// Standard
	"math"
)

const (
	// HTK Mel Scale Constants (Matches Librosa defaults)
	MelScalar = 2595.0
	MelBreak  = 700.0
)

// Converts a Linear Power Spectrogram to a Mel Spectrogram; Replicates librosa.filters.mel()
func applyMelFilterbank(powerSpec [][]float64, numLinearBins, numFrames int) [][]float64 {
	minMel := hzToMel(MinFreq)
	maxMel := hzToMel(MaxFreq)
	
	// Create Mel Frequency Points
	melPoints := make([]float64, Nmels+2)
	step := (maxMel - minMel) / float64(Nmels+1)

	for i := 0; i < len(melPoints); i++ {
		melPoints[i] = melToHz(minMel + float64(i)*step)
	}

	// Convert Hz to FFT Bin Indices
	binPoints := make([]int, len(melPoints))
	for i, freq := range melPoints {
		// bin = freq * (Nfft+1) / SampleRate
		binPoints[i] = int(math.Floor((Nfft + 1) * freq / float64(ffmpeg.SampleRate)))
	}

	// Create Filter Matrix [Nmels][LinearBins]
	filters := make([][]float64, Nmels)
	for i := 0; i < Nmels; i++ {
		filters[i] = make([]float64, numLinearBins)
		start := binPoints[i]
		center := binPoints[i+1]
		end := binPoints[i+2]

		// Triangle Up
		for f := start; f < center; f++ {
			if f >= numLinearBins { break }
			filters[i][f] = float64(f-start) / float64(center-start)
		}
		// Triangle Down
		for f := center; f < end; f++ {
			if f >= numLinearBins { break }
			filters[i][f] = float64(end-f) / float64(end-center)
		}
	}

	// Matrix Dot Product: [Nmels x Linear] . [Linear x Time] = [Nmels x Time]
	melSpec := make([][]float64, Nmels)
	for i := range melSpec {
		melSpec[i] = make([]float64, numFrames)
	}

	for m := 0; m < Nmels; m++ {
		for t := 0; t < numFrames; t++ {
			sum := 0.0
			for k := 0; k < numLinearBins; k++ {
				sum += filters[m][k] * powerSpec[k][t]
			}
			melSpec[m][t] = sum
		}
	}
	return melSpec
}

// Clips DB to [-80, 0] and scales to [0.0, 1.0]
func fixedNormalize(dbSpec [][]float64) [][]float64 {
	rows := len(dbSpec)
	cols := len(dbSpec[0])
	norm := make([][]float64, rows)

	for r := 0; r < rows; r++ {
		norm[r] = make([]float64, cols)
		for c := 0; c < cols; c++ {
			val := dbSpec[r][c]
			if val < -80.0 { val = -80.0 }
			if val > 0.0 { val = 0.0 }
			norm[r][c] = (val + 80.0) / 80.0
		}
	}
	return norm
}

// Converts a power spectrogram to decibels
func powerToDb(spec [][]float64) [][]float64 {
	rows := len(spec)
	cols := len(spec[0])
	dbSpec := make([][]float64, rows)
	
	// Find Global Max for Reference
	var maxVal float64 = 1e-10
	for r := 0; r < rows; r++ {
		dbSpec[r] = make([]float64, cols)
		for c := 0; c < cols; c++ {
			if spec[r][c] > maxVal { maxVal = spec[r][c] }
		}
	}

	// Log Calculation
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			val := spec[r][c]
			if val < 1e-10 { val = 1e-10 }
			dbSpec[r][c] = 10.0 * math.Log10(val/maxVal)
		}
	}
	return dbSpec
}

// Internal Helper: Convert Hz to Mel
func hzToMel(hz float64) float64 {
	return MelScalar * math.Log10(1.0+hz/MelBreak)
}

// Internal Helper: Convert Mel to Hz
func melToHz(mel float64) float64 {
	return MelBreak * (math.Pow(10, mel/MelScalar) - 1.0)
}
