package whisper

import (
	"strconv"
	"strings"
	"time"
)

// Segment represents a transcribed segment with timing information.
type Segment struct {
	// Text is the transcribed text for this segment.
	Text string

	// Start is the start time of this segment.
	Start time.Duration

	// End is the end time of this segment.
	End time.Duration

	// Score is the confidence score (if ReturnScores was enabled).
	Score float32

	// NoSpeechProb is the no-speech probability (if ReturnNoSpeechProb was enabled).
	NoSpeechProb float32
}

// Result contains the complete transcription result.
type Result struct {
	// Language is the detected or specified language code.
	Language string

	// LanguageProbability is the confidence of language detection.
	LanguageProbability float32

	// Segments contains the transcribed segments with timing.
	Segments []Segment

	// Text is the full transcription text (convenience field).
	// This is the concatenation of all segment texts.
	Text string

	// Duration is the total audio duration.
	Duration time.Duration
}

// LanguageProb represents a language detection result.
type LanguageProb struct {
	// Language is the ISO 639-1 language code.
	Language string

	// Probability is the detection confidence (0.0 to 1.0).
	Probability float32
}

// String returns the full transcription text.
func (r *Result) String() string {
	return r.Text
}

// SRT returns the transcription in SRT subtitle format.
func (r *Result) SRT() string {
	var sb strings.Builder
	for i, seg := range r.Segments {
		sb.WriteString(formatSRTEntry(i+1, seg.Start, seg.End, seg.Text))
	}
	return sb.String()
}

// VTT returns the transcription in WebVTT subtitle format.
func (r *Result) VTT() string {
	var sb strings.Builder
	sb.WriteString("WEBVTT\n\n")
	for _, seg := range r.Segments {
		sb.WriteString(formatVTTEntry(seg.Start, seg.End, seg.Text))
	}
	return sb.String()
}

// formatSRTEntry formats a single SRT entry.
func formatSRTEntry(index int, start, end time.Duration, text string) string {
	var sb strings.Builder
	sb.WriteString(strconv.Itoa(index))
	sb.WriteString("\n")
	sb.WriteString(formatSRTTime(start))
	sb.WriteString(" --> ")
	sb.WriteString(formatSRTTime(end))
	sb.WriteString("\n")
	sb.WriteString(text)
	sb.WriteString("\n\n")
	return sb.String()
}

// formatVTTEntry formats a single WebVTT entry.
func formatVTTEntry(start, end time.Duration, text string) string {
	var sb strings.Builder
	sb.WriteString(formatVTTTime(start))
	sb.WriteString(" --> ")
	sb.WriteString(formatVTTTime(end))
	sb.WriteString("\n")
	sb.WriteString(text)
	sb.WriteString("\n\n")
	return sb.String()
}

// formatSRTTime formats a duration as SRT timestamp (HH:MM:SS,mmm).
func formatSRTTime(d time.Duration) string {
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second
	d -= s * time.Second
	ms := d / time.Millisecond

	return formatTimeComponents(int(h), int(m), int(s), int(ms), ",")
}

// formatVTTTime formats a duration as WebVTT timestamp (HH:MM:SS.mmm).
func formatVTTTime(d time.Duration) string {
	h := d / time.Hour
	d -= h * time.Hour
	m := d / time.Minute
	d -= m * time.Minute
	s := d / time.Second
	d -= s * time.Second
	ms := d / time.Millisecond

	return formatTimeComponents(int(h), int(m), int(s), int(ms), ".")
}

// formatTimeComponents formats time components as HH:MM:SS<sep>mmm.
func formatTimeComponents(h, m, s, ms int, sep string) string {
	return formatPadded(h, 2) + ":" +
		formatPadded(m, 2) + ":" +
		formatPadded(s, 2) + sep +
		formatPadded(ms, 3)
}

// formatPadded formats an integer with zero padding.
func formatPadded(n, width int) string {
	s := strconv.Itoa(n)
	for len(s) < width {
		s = "0" + s
	}
	return s
}
