package ingestion

import (
	"fmt"
	"path/filepath"
	"strings"
)

type Document struct {
	Content string
	Source  string
	Pages   []Page
}

type Page struct {
	Number  int
	Content string
}

type Loader interface {
	Load(path string) (*Document, error)
	Supports(ext string) bool
}

type Router struct {
	loaders []Loader
}

func NewRouter() *Router {
	return NewRouterWithLoaders(
		&TextLoader{},
		&PdfLoader{},
		&MarkdownLoader{},
	)
}

func NewRouterWithLoaders(loaders ...Loader) *Router {
	router := &Router{
		loaders: make([]Loader, 0, len(loaders)),
	}
	for _, loader := range loaders {
		if loader != nil {
			router.loaders = append(router.loaders, loader)
		}
	}

	return router
}

func (r *Router) Load(path string) (*Document, error) {
	ext := strings.ToLower(filepath.Ext(path))
	if ext == "" {
		return nil, fmt.Errorf("missing file extension for %q", path)
	}

	for _, loader := range r.loaders {
		if loader.Supports(ext) {
			return loader.Load(path)
		}
	}

	return nil, fmt.Errorf("unsupported file extension %q", ext)
}
