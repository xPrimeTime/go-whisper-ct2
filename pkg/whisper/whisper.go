package whisper

/*
#cgo CFLAGS: -I${SRCDIR}/../../csrc/include
#cgo LDFLAGS: -L${SRCDIR}/../../csrc/build -lwhisper_ct2 -lstdc++ -lm

#include <whisper_ct2.h>
#include <stdlib.h>
*/
import "C"

// Version returns the library version string.
func Version() string {
	return C.GoString(C.whisper_ct2_version())
}

// CTranslate2Version returns the CTranslate2 version string.
func CTranslate2Version() string {
	return C.GoString(C.whisper_ct2_ctranslate2_version())
}

// SupportedAudioFormats returns a comma-separated list of supported audio formats.
func SupportedAudioFormats() string {
	return C.GoString(C.whisper_ct2_supported_audio_formats())
}
