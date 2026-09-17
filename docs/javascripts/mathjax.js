window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex"
  }
};

// Instant navigation (`navigation.instant`) swaps the page content without a
// page load, so MathJax has to typeset again on every page change. `document$`
// is the observable of the theme which emits the document once for the
// initial page and then on every navigation. MathJax typesets the initial
// page on its own when it loads, so the first emission is skipped; typesetting
// it a second time would duplicate every formula.
var mathjaxInitialPage = true;
document$.subscribe(function () {
  if (mathjaxInitialPage) {
    mathjaxInitialPage = false;
    return;
  }
  if (window.MathJax && typeof MathJax.typesetPromise === "function") {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    MathJax.typesetPromise();
  }
});
