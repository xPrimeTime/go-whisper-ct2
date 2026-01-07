// Package whisper provides Go bindings to CTranslate2's Whisper speech recognition.
//
// This package enables high-quality speech-to-text transcription using OpenAI's
// Whisper models converted to CTranslate2 format, without requiring Python.
//
// # Basic Usage
//
//	// Load a model
//	model, err := whisper.LoadModel("./whisper-base-ct2", whisper.DefaultModelConfig())
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer model.Close()
//
//	// Transcribe an audio file
//	result, err := model.TranscribeFile("audio.wav", whisper.WithLanguage("en"))
//	if err != nil {
//		log.Fatal(err)
//	}
//
//	fmt.Println(result.Text)
//
// # Model Conversion
//
// Whisper models must be converted to CTranslate2 format before use:
//
//	pip install ctranslate2 transformers[torch]
//	ct2-transformers-converter --model openai/whisper-base \
//	    --output_dir whisper-base-ct2 --quantization float32
//
// # Transcription Options
//
// Transcription behavior can be customized using functional options:
//
//	result, err := model.TranscribeFile("audio.wav",
//	    whisper.WithLanguage("en"),
//	    whisper.WithTask("transcribe"),
//	    whisper.WithBeamSize(5),
//	)
//
// # Language Detection
//
// For multilingual models, language can be auto-detected:
//
//	result, err := model.TranscribeFile("audio.wav", whisper.WithLanguage("auto"))
//	fmt.Println("Detected language:", result.Language)
//
// Or detected explicitly:
//
//	probs, err := model.DetectLanguage("audio.wav")
//	fmt.Println("Top language:", probs[0].Language, probs[0].Probability)
//
// # Thread Safety
//
// Model instances are safe for concurrent use from multiple goroutines.
// Each transcription call uses its own internal state.
package whisper
