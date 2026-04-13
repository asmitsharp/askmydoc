package ingestion

import (
	"strings"
	"testing"
)

type fakeLoader struct {
	ext    string
	called bool
}

func (f *fakeLoader) Supports(ext string) bool {
	return ext == f.ext
}

func (f *fakeLoader) Load(path string) (*Document, error) {
	f.called = true
	return &Document{Content: path}, nil
}

func TestRouterLoadRoutesByExtension(t *testing.T) {
	loader := &fakeLoader{ext: ".md"}
	router := NewRouterWithLoaders(loader)

	doc, err := router.Load("README.MD")
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if !loader.called {
		t.Fatal("expected fake loader to be called")
	}
	if doc.Content != "README.MD" {
		t.Fatalf("Content = %q, want %q", doc.Content, "README.MD")
	}
}

func TestRouterLoadRejectsUnsupportedExtension(t *testing.T) {
	router := NewRouterWithLoaders()

	_, err := router.Load("data.csv")
	if err == nil {
		t.Fatal("expected unsupported extension error")
	}
	if !strings.Contains(err.Error(), ".csv") {
		t.Fatalf("error = %q, want it to include extension", err)
	}
}

func TestRouterLoadRejectsMissingExtension(t *testing.T) {
	router := NewRouterWithLoaders(&fakeLoader{ext: ".txt"})

	_, err := router.Load("README")
	if err == nil {
		t.Fatal("expected missing extension error")
	}
	if !strings.Contains(err.Error(), "missing file extension") {
		t.Fatalf("error = %q, want missing extension message", err)
	}
}
