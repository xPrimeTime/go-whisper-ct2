package whisper

/*
#include <whisper_ct2.h>
*/
import "C"
import (
	"errors"
	"fmt"
)

// Sentinel errors for common failure modes.
var (
	ErrInvalidArgument          = errors.New("whisper: invalid argument")
	ErrModelNotFound            = errors.New("whisper: model not found")
	ErrModelLoadFailed          = errors.New("whisper: failed to load model")
	ErrAudioLoadFailed          = errors.New("whisper: failed to load audio")
	ErrTranscribeFailed         = errors.New("whisper: transcription failed")
	ErrOutOfMemory              = errors.New("whisper: out of memory")
	ErrInvalidModel             = errors.New("whisper: invalid model handle")
	ErrNotMultilingual          = errors.New("whisper: model is not multilingual")
	ErrUnsupportedAudioFormat   = errors.New("whisper: unsupported audio format")
	ErrEmptyAudio               = errors.New("whisper: empty audio input")
	ErrInternal                 = errors.New("whisper: internal error")
)

// Error wraps a whisper error with additional context from the C layer.
type Error struct {
	Code    int
	Message string
	Err     error
}

func (e *Error) Error() string {
	if e.Message != "" {
		return fmt.Sprintf("%v: %s", e.Err, e.Message)
	}
	return e.Err.Error()
}

func (e *Error) Unwrap() error {
	return e.Err
}

// errorFromCode converts a C error code to a Go error.
func errorFromCode(code C.whisper_error_t) error {
	if code == C.WHISPER_OK {
		return nil
	}

	var baseErr error
	switch code {
	case C.WHISPER_ERROR_INVALID_ARGUMENT:
		baseErr = ErrInvalidArgument
	case C.WHISPER_ERROR_MODEL_NOT_FOUND:
		baseErr = ErrModelNotFound
	case C.WHISPER_ERROR_MODEL_LOAD_FAILED:
		baseErr = ErrModelLoadFailed
	case C.WHISPER_ERROR_AUDIO_LOAD_FAILED:
		baseErr = ErrAudioLoadFailed
	case C.WHISPER_ERROR_TRANSCRIPTION_FAILED:
		baseErr = ErrTranscribeFailed
	case C.WHISPER_ERROR_OUT_OF_MEMORY:
		baseErr = ErrOutOfMemory
	case C.WHISPER_ERROR_INVALID_MODEL:
		baseErr = ErrInvalidModel
	case C.WHISPER_ERROR_NOT_MULTILINGUAL:
		baseErr = ErrNotMultilingual
	case C.WHISPER_ERROR_UNSUPPORTED_AUDIO_FORMAT:
		baseErr = ErrUnsupportedAudioFormat
	default:
		baseErr = ErrInternal
	}

	// Get detailed message from thread-local storage
	msg := C.GoString(C.whisper_get_last_error_message())
	C.whisper_clear_error()

	if msg != "" {
		return &Error{
			Code:    int(code),
			Message: msg,
			Err:     baseErr,
		}
	}
	return baseErr
}
