pdf.js 4.0.379 (`pdfjs-dist`, Apache-2.0), vendored so the viewer works
without a third-party CDN and under a strict CSP.

- `pdf.min.js`        = `package/build/pdf.min.mjs`
- `pdf.worker.min.js` = `package/build/pdf.worker.min.mjs`

Renamed to `.js` so Streamlit's static file server sends a JavaScript MIME
type for the ES-module import.  Source tarball integrity (npm):
`sha512-6H0Gv1nna+wmrr3CakaKlZ4rbrL8hvGIFAgg4YcoFuGC0HC4B2DVjXEGTFjJEjLlf8nYi3C3/MYRcM5bNx0elA==`
