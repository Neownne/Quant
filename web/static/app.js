function initChart(elId, option) {
    const el = document.getElementById(elId);
    if (!el) return;
    const chart = echarts.init(el, 'dark');
    chart.setOption(option);
    window.addEventListener('resize', () => chart.resize());
    return chart;
}

document.body.addEventListener('htmx:afterSettle', function() {
    document.querySelectorAll('[data-chart]').forEach(el => {
        const id = el.id;
        try {
            const spec = JSON.parse(el.dataset.chart);
            initChart(id, spec);
        } catch(e) {}
    });
});
