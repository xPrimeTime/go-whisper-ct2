package whisper

/*
#include <whisper_ct2.h>
#include <stdlib.h>
*/
import "C"
import (
	"runtime"
	"unsafe"
)

// Model represents a loaded Whisper model.
type Model struct {
	handle C.whisper_model_t
}

// ModelConfig contains model loading configuration.
type ModelConfig struct {
	// Device specifies the compute device. Currently only "cpu" is supported.
	Device string

	// ComputeType specifies the numerical precision.
	// Options: "int8", "int16", "float16", "float32", "default"
	ComputeType string

	// InterThreads specifies the number of threads for batch parallelization.
	// Default: 1
	InterThreads int

	// IntraThreads specifies the number of threads within operations.
	// 0 means auto-detect based on CPU cores.
	// Default: 0
	IntraThreads int
}

// DefaultModelConfig returns sensible defaults for model configuration.
func DefaultModelConfig() ModelConfig {
	return ModelConfig{
		Device:       "cpu",
		ComputeType:  "default",
		InterThreads: 1,
		IntraThreads: 0,
	}
}

// LoadModel loads a Whisper model from a CTranslate2 model directory.
//
// The model directory should contain files created by ct2-transformers-converter,
// including model.bin, vocabulary files, and configuration.
//
// Example:
//
//	model, err := whisper.LoadModel("./whisper-base-ct2", whisper.DefaultModelConfig())
//	if err != nil {
//		return err
//	}
//	defer model.Close()
func LoadModel(path string, config ModelConfig) (*Model, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var cConfig C.whisper_model_config_t
	C.whisper_model_config_init(&cConfig)

	if config.Device != "" {
		cDevice := C.CString(config.Device)
		defer C.free(unsafe.Pointer(cDevice))
		cConfig.device = cDevice
	}

	if config.ComputeType != "" {
		cComputeType := C.CString(config.ComputeType)
		defer C.free(unsafe.Pointer(cComputeType))
		cConfig.compute_type = cComputeType
	}

	cConfig.inter_threads = C.int32_t(config.InterThreads)
	cConfig.intra_threads = C.int32_t(config.IntraThreads)

	var handle C.whisper_model_t
	errCode := C.whisper_model_load(cPath, &cConfig, &handle)
	if errCode != C.WHISPER_OK {
		return nil, errorFromCode(errCode)
	}

	model := &Model{handle: handle}
	runtime.SetFinalizer(model, (*Model).Close)
	return model, nil
}

// Close releases model resources.
//
// It's safe to call Close multiple times. After Close is called,
// the model should not be used for transcription.
func (m *Model) Close() error {
	if m.handle != nil {
		C.whisper_model_free(m.handle)
		m.handle = nil
	}
	runtime.SetFinalizer(m, nil)
	return nil
}

// IsMultilingual returns true if the model supports multiple languages.
//
// Multilingual models can detect language automatically and translate
// between languages. English-only models (like whisper-base.en) return false.
func (m *Model) IsMultilingual() bool {
	if m.handle == nil {
		return false
	}
	return bool(C.whisper_model_is_multilingual(m.handle))
}

// NumMels returns the number of mel frequency bins used by the model.
//
// Standard Whisper models use 80 mel bins. The large-v3 model uses 128.
func (m *Model) NumMels() int {
	if m.handle == nil {
		return 0
	}
	return int(C.whisper_model_n_mels(m.handle))
}
