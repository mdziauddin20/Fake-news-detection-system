/* ============================================================
   TruthScan — Fake News Detection System
   Client-side interactions
   ============================================================ */

// ---- Word / character counter ----
const textarea    = document.getElementById('newsInput');
const wordCounter = document.getElementById('wordCounter');

function updateCounter() {
    if (!textarea || !wordCounter) return;
    const text  = textarea.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    wordCounter.textContent = words.toLocaleString() + (words === 1 ? ' word' : ' words');
}

if (textarea) {
    textarea.addEventListener('input', updateCounter);
    updateCounter(); // initialise on page load (populated on result page)
}

// ---- Loading state on form submit ----
const form    = document.getElementById('newsForm');
const btn     = document.getElementById('analyzeBtn');

if (form && btn) {
    form.addEventListener('submit', function () {
        const btnText    = btn.querySelector('.btn-text');
        const btnLoading = btn.querySelector('.btn-loading');

        if (btnText)    btnText.style.display    = 'none';
        if (btnLoading) btnLoading.style.display = 'inline-flex';
        btn.disabled = true;
    });
}

// ---- Auto-scroll to results after submission ----
const resultsSection = document.getElementById('results');
if (resultsSection) {
    setTimeout(function () {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 150);
}
