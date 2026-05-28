// Track chart instances to dispose before replacement
const _charts = {};

function initChart(el) {
    const id = el.id;
    if (!id) {
        console.warn('Chart element missing id');
        return;
    }
    if (typeof echarts === 'undefined') {
        console.error('ECharts library not loaded');
        return;
    }
    if (_charts[id]) {
        _charts[id].dispose();
        delete _charts[id];
    }
    // Ensure element has dimensions before init
    if (el.offsetWidth === 0 || el.offsetHeight === 0) {
        console.warn('Chart element has zero dimensions:', id, el.offsetWidth, el.offsetHeight);
        return;
    }
    try {
        const raw = el.getAttribute('data-chart');
        const option = JSON.parse(raw);
        const chart = echarts.init(el);
        chart.setOption(option);
        _charts[id] = chart;
    } catch(e) {
        console.error('ECharts init error for', id, ':', e);
        el.textContent = '图表加载失败: ' + e.message;
    }
}

function refreshCharts() {
    document.querySelectorAll('[data-chart]').forEach(function(el) {
        if (!_charts[el.id]) {
            initChart(el);
        }
    });
}

// Listen for HTMX content swaps
document.addEventListener('htmx:afterSettle', function() {
    // Small delay to ensure DOM layout is complete
    setTimeout(refreshCharts, 50);
});

// Also init any charts present on initial page load
document.addEventListener('DOMContentLoaded', function() {
    setTimeout(refreshCharts, 100);
});

window.addEventListener('resize', function() {
    Object.values(_charts).forEach(function(c) { c.resize(); });
});
