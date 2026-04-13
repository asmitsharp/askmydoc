package ingestion

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"unicode"

	"github.com/gen2brain/go-fitz"
	"github.com/ledongthuc/pdf"
)

const (
	minEmbeddedTextRunes = 40
	ocrDPI               = 300.0
	ocrLanguage          = "eng"
	ocrPageSegMode       = "6"
)

type PdfLoader struct{}

func (p *PdfLoader) Supports(ext string) bool {
	return ext == ".pdf"
}

func (p *PdfLoader) Load(path string) (*Document, error) {
	doc, embeddedErr := loadEmbeddedPDFText(path)
	if embeddedErr == nil && hasEnoughText(doc) {
		return doc, nil
	}

	ocrDoc, ocrErr := loadOCRPDFText(path)
	if ocrErr != nil {
		if embeddedErr != nil {
			return nil, fmt.Errorf("embedded pdf extraction failed: %w; ocr fallback failed: %w", embeddedErr, ocrErr)
		}
		return nil, fmt.Errorf("pdf contains too little embedded text; ocr fallback failed: %w", ocrErr)
	}

	return ocrDoc, nil
}

func loadEmbeddedPDFText(path string) (*Document, error) {
	file, reader, err := pdf.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open pdf: %w", err)
	}
	defer file.Close()

	pages := make([]Page, 0, reader.NumPage())

	for pageNumber := 1; pageNumber <= reader.NumPage(); pageNumber++ {
		page := reader.Page(pageNumber)
		if page.V.IsNull() || page.V.Key("Contents").Kind() == pdf.Null {
			continue
		}

		pageText, err := extractPageText(page)
		if err != nil {
			return nil, fmt.Errorf("extract page %d: %w", pageNumber, err)
		}
		if pageText == "" {
			continue
		}

		pages = append(pages, Page{
			Number:  pageNumber,
			Content: pageText,
		})
	}

	return newDocumentFromPages(pages), nil
}

func extractPageText(page pdf.Page) (string, error) {
	rows, err := page.GetTextByRow()
	if err != nil {
		return "", err
	}

	lines := make([]string, 0, len(rows))
	for _, row := range rows {
		line := normalizeTextLine(row.Content)
		if line != "" {
			lines = append(lines, line)
		}
	}

	return strings.Join(lines, "\n"), nil
}

func loadOCRPDFText(path string) (*Document, error) {
	tesseractPath, err := exec.LookPath("tesseract")
	if err != nil {
		return nil, fmt.Errorf("tesseract is required for scanned PDFs: install it with `brew install tesseract`: %w", err)
	}

	doc, err := fitz.New(path)
	if err != nil {
		return nil, fmt.Errorf("open pdf renderer: %w", err)
	}
	defer doc.Close()

	pages := make([]Page, 0, doc.NumPage())
	for pageIndex := 0; pageIndex < doc.NumPage(); pageIndex++ {
		png, err := doc.ImagePNG(pageIndex, ocrDPI)
		if err != nil {
			return nil, fmt.Errorf("render page %d: %w", pageIndex+1, err)
		}

		pageText, err := runTesseractOCR(tesseractPath, png)
		if err != nil {
			return nil, fmt.Errorf("ocr page %d: %w", pageIndex+1, err)
		}

		pageText = normalizeOCRText(pageText)
		if pageText == "" {
			continue
		}

		pages = append(pages, Page{
			Number:  pageIndex + 1,
			Content: pageText,
		})
	}

	if len(pages) == 0 {
		return nil, fmt.Errorf("ocr produced no extractable text")
	}

	return newDocumentFromPages(pages), nil
}

func runTesseractOCR(tesseractPath string, png []byte) (string, error) {
	imageFile, err := os.CreateTemp("", "askmydoc-ocr-*.png")
	if err != nil {
		return "", fmt.Errorf("create temp image: %w", err)
	}
	imagePath := imageFile.Name()
	defer os.Remove(imagePath)

	if _, err := imageFile.Write(png); err != nil {
		imageFile.Close()
		return "", fmt.Errorf("write temp image: %w", err)
	}
	if err := imageFile.Close(); err != nil {
		return "", fmt.Errorf("close temp image: %w", err)
	}

	cmd := exec.Command(tesseractPath, imagePath, "stdout", "--psm", ocrPageSegMode, "-l", ocrLanguage)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	out, err := cmd.Output()
	if err != nil {
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return "", fmt.Errorf("%w: %s", err, msg)
		}
		return "", err
	}

	return string(out), nil
}

func normalizeTextLine(words pdf.TextHorizontal) string {
	var line strings.Builder
	for _, word := range words {
		text := strings.TrimSpace(word.S)
		if text == "" {
			continue
		}

		if line.Len() > 0 && !strings.HasSuffix(line.String(), " ") {
			line.WriteByte(' ')
		}
		line.WriteString(text)
	}

	return strings.TrimSpace(line.String())
}

func normalizeOCRText(text string) string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	text = strings.ReplaceAll(text, "\r", "\n")

	lines := strings.Split(text, "\n")
	normalized := make([]string, 0, len(lines))
	previousBlank := false

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			if len(normalized) > 0 && !previousBlank {
				normalized = append(normalized, "")
				previousBlank = true
			}
			continue
		}

		normalized = append(normalized, line)
		previousBlank = false
	}

	for len(normalized) > 0 && normalized[len(normalized)-1] == "" {
		normalized = normalized[:len(normalized)-1]
	}

	return strings.Join(normalized, "\n")
}

func newDocumentFromPages(pages []Page) *Document {
	var content bytes.Buffer
	for _, page := range pages {
		if content.Len() > 0 {
			content.WriteString("\n\n")
		}
		content.WriteString(fmt.Sprintf("[Page %d]\n%s", page.Number, page.Content))
	}

	return &Document{
		Content: content.String(),
		Pages:   pages,
	}
}

func hasEnoughText(doc *Document) bool {
	if doc == nil {
		return false
	}

	count := 0
	for _, r := range doc.Content {
		if !unicode.IsSpace(r) {
			count++
		}
		if count >= minEmbeddedTextRunes {
			return true
		}
	}

	return false
}
