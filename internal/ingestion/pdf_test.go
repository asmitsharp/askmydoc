package ingestion

import "testing"

func TestNormalizeOCRText(t *testing.T) {
	input := "  first line  \r\n\r\n\r\n second line \n\n third line \r\n"
	got := normalizeOCRText(input)
	want := "first line\n\nsecond line\n\nthird line"

	if got != want {
		t.Fatalf("normalizeOCRText() = %q, want %q", got, want)
	}
}

func TestNewDocumentFromPages(t *testing.T) {
	doc := newDocumentFromPages([]Page{
		{Number: 1, Content: "alpha"},
		{Number: 2, Content: "beta"},
	})

	want := "[Page 1]\nalpha\n\n[Page 2]\nbeta"
	if doc.Content != want {
		t.Fatalf("Content = %q, want %q", doc.Content, want)
	}
	if len(doc.Pages) != 2 {
		t.Fatalf("len(Pages) = %d, want 2", len(doc.Pages))
	}
}
