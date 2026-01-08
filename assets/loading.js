// Loading overlay control for Sentiment Game Analytics Dashboard
// Uses MutationObserver to detect when Plotly charts finish rendering

// Global flags
window.loadingActive = false;
window.chartObserver = null;
window.lastDropdownValue = {};

// Dash clientside callbacks
window.dash_clientside = Object.assign({}, window.dash_clientside, {
    loading: {
        showLoading: function () {
            showLoadingOverlay();
            return window.dash_clientside.no_update;
        },
        hideLoading: function () {
            hideLoadingOverlay();
            return window.dash_clientside.no_update;
        }
    }
});

// Show loading overlay
function showLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        window.loadingActive = true;
        overlay.classList.remove('hidden', 'fade-out');

        // Start watching for chart renders
        startChartObserver();
    }
}

// Hide loading overlay with fade
function hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.add('fade-out');
        setTimeout(() => {
            overlay.classList.add('hidden');
            window.loadingActive = false;
        }, 300);
    }
}

// Watch for Plotly charts to finish rendering
function startChartObserver() {
    // Stop any existing observer
    if (window.chartObserver) {
        window.chartObserver.disconnect();
    }

    // Set a minimum display time
    const minDisplayTime = 600;
    const startTime = Date.now();

    // Create observer to watch for Plotly SVG elements
    window.chartObserver = new MutationObserver((mutations) => {
        // Check if we have Plotly charts rendered
        const charts = document.querySelectorAll('.js-plotly-plot .main-svg');
        const tabContent = document.getElementById('tab-content');

        if (charts.length > 0 && tabContent) {
            // Charts found - wait for minimum display time, then hide overlay
            const elapsed = Date.now() - startTime;
            const remainingTime = Math.max(0, minDisplayTime - elapsed);

            setTimeout(() => {
                hideLoadingOverlay();
                if (window.chartObserver) {
                    window.chartObserver.disconnect();
                    window.chartObserver = null;
                }
            }, remainingTime);
        }
    });

    // Observe the entire document for changes
    window.chartObserver.observe(document.body, {
        childList: true,
        subtree: true
    });

    // Fallback: hide after max time in case observer misses
    setTimeout(() => {
        if (window.loadingActive) {
            hideLoadingOverlay();
            if (window.chartObserver) {
                window.chartObserver.disconnect();
                window.chartObserver = null;
            }
        }
    }, 2500);
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function () {

    // 1. BEGIN button click
    document.body.addEventListener('click', function (e) {
        if (e.target.closest('#begin-btn')) {
            showLoadingOverlay();
        }
    }, true);  // Use capture phase

    // 2. Tab clicks - detect all 5 tabs
    document.body.addEventListener('click', function (e) {
        // Check for tab element with multiple selectors
        const tab = e.target.closest('.tab');
        const customTab = e.target.closest('.custom-tabs .tab');
        const tabDiv = e.target.closest('[class*="Tab"]');

        // Get the clicked element or its parent with tab class
        const clickedTab = tab || customTab || tabDiv;

        if (clickedTab) {
            // Check if not already selected
            const isSelected = clickedTab.classList.contains('tab--selected') ||
                clickedTab.getAttribute('aria-selected') === 'true';

            if (!isSelected) {
                showLoadingOverlay();
            }
        }
    }, true);  // Use capture phase for earlier detection

    // 3. Watch for dropdown value changes using polling
    setInterval(() => {
        const dropdowns = document.querySelectorAll('.Select-value-label');
        dropdowns.forEach((dropdown, index) => {
            const currentValue = dropdown.textContent;
            const key = `dropdown_${index}`;

            if (window.lastDropdownValue[key] &&
                window.lastDropdownValue[key] !== currentValue &&
                !window.loadingActive) {
                showLoadingOverlay();
            }

            window.lastDropdownValue[key] = currentValue;
        });
    }, 100);
});

// Global tab click handler as backup
document.addEventListener('click', function (e) {
    const tab = e.target.closest('.tab');
    if (tab && !tab.classList.contains('tab--selected')) {
        showLoadingOverlay();
    }
}, true);
