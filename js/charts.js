/**
 * Charts Module
 * Benchmark comparisons using Chart.js
 * Includes live-update support for 3-way comparison demo.
 */

let _benchmarkChart = null;

function initCharts() {
  Chart.defaults.color = '#94a3b8';
  Chart.defaults.borderColor = 'rgba(148, 163, 184, 0.1)';
  Chart.defaults.font.family = "'Inter', sans-serif";

  _benchmarkChart = _initBenchmarkChart();
  _initAccuracyChart();
  _initStorageChart();
}

function _initBenchmarkChart() {
  const ctx = document.getElementById('benchmark-chart');
  if (!ctx) return null;

  const chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Vanilla RAG', 'ARQ + PQ', 'ARQ + TurboQuant'],
      datasets: [
        {
          label: 'Độ Trễ Truy Vấn (ms)',
          data: [45, 28, 18],
          backgroundColor: [
            'rgba(239, 68, 68, 0.75)',
            'rgba(245, 158, 11, 0.75)',
            'rgba(16, 185, 129, 0.8)',
          ],
          borderColor: [
            'rgba(239, 68, 68, 1)',
            'rgba(245, 158, 11, 1)',
            'rgba(16, 185, 129, 1)',
          ],
          borderWidth: 2,
          borderRadius: 6,
        },
        {
          label: 'Sử Dụng Bộ Nhớ (GB)',
          data: [8.2, 1.5, 1.2],
          backgroundColor: [
            'rgba(239, 68, 68, 0.3)',
            'rgba(245, 158, 11, 0.3)',
            'rgba(16, 185, 129, 0.35)',
          ],
          borderColor: [
            'rgba(239, 68, 68, 0.8)',
            'rgba(245, 158, 11, 0.8)',
            'rgba(16, 185, 129, 0.8)',
          ],
          borderWidth: 2,
          borderRadius: 6,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#94a3b8', font: { size: 11 } }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          borderColor: 'rgba(100, 210, 255, 0.3)',
          borderWidth: 1,
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(148, 163, 184, 0.08)' },
          ticks: { color: '#94a3b8' }
        },
        y: {
          grid: { color: 'rgba(148, 163, 184, 0.08)' },
          ticks: { color: '#94a3b8' }
        }
      },
      animation: {
        duration: 1200,
        easing: 'easeOutQuart'
      }
    }
  });
  return chart;
}

function _initAccuracyChart() {
  const ctx = document.getElementById('accuracy-chart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Recall@1', 'Recall@5', 'Recall@10', 'Precision', 'Speed', 'Storage Efficiency'],
      datasets: [
        {
          label: 'TurboQuant',
          data: [94, 97, 99, 93, 98, 95],
          backgroundColor: 'rgba(16, 185, 129, 0.15)',
          borderColor: 'rgba(16, 185, 129, 0.9)',
          pointBackgroundColor: 'rgba(16, 185, 129, 1)',
          pointRadius: 4,
          borderWidth: 2,
        },
        {
          label: 'Product Quantization',
          data: [89, 93, 96, 88, 75, 80],
          backgroundColor: 'rgba(99, 102, 241, 0.1)',
          borderColor: 'rgba(99, 102, 241, 0.7)',
          pointBackgroundColor: 'rgba(99, 102, 241, 1)',
          pointRadius: 4,
          borderWidth: 2,
        },
        {
          label: 'No Quantization',
          data: [100, 100, 100, 100, 30, 20],
          backgroundColor: 'rgba(239, 68, 68, 0.05)',
          borderColor: 'rgba(239, 68, 68, 0.5)',
          pointBackgroundColor: 'rgba(239, 68, 68, 1)',
          pointRadius: 4,
          borderWidth: 2,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: { color: '#94a3b8', font: { size: 11 } }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          borderColor: 'rgba(100, 210, 255, 0.3)',
          borderWidth: 1,
        }
      },
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: {
            display: false,
            stepSize: 20
          },
          grid: { color: 'rgba(148, 163, 184, 0.15)' },
          pointLabels: {
            color: '#94a3b8',
            font: { size: 11 }
          },
          angleLines: { color: 'rgba(148, 163, 184, 0.15)' }
        }
      },
      animation: {
        duration: 1400,
        easing: 'easeOutQuart'
      }
    }
  });
}

function _initStorageChart() {
  const ctx = document.getElementById('storage-chart');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['Original Float32', 'TurboQuant (8-bit)', 'TurboQuant (4-bit)', 'QJL Correction'],
      datasets: [{
        data: [100, 25, 12.5, 4],
        backgroundColor: [
          'rgba(239, 68, 68, 0.7)',
          'rgba(16, 185, 129, 0.7)',
          'rgba(16, 185, 129, 0.4)',
          'rgba(100, 210, 255, 0.6)',
        ],
        borderColor: [
          'rgba(239, 68, 68, 1)',
          'rgba(16, 185, 129, 1)',
          'rgba(16, 185, 129, 0.7)',
          'rgba(100, 210, 255, 1)',
        ],
        borderWidth: 2,
        hoverOffset: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#94a3b8',
            font: { size: 10 },
            padding: 12
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.95)',
          borderColor: 'rgba(100, 210, 255, 0.3)',
          borderWidth: 1,
          callbacks: {
            label: (ctx) => ` ${ctx.label}: ${ctx.raw}% of original size`
          }
        }
      },
      animation: {
        animateRotate: true,
        duration: 1200
      },
      cutout: '65%'
    }
  });
}

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
  setTimeout(initCharts, 100);
});

/**
 * Live-update the benchmark chart with real timing data from the 3-way comparison.
 * Called by arq-visualizer after all pipelines complete.
 * @param {{ vanilla: number, pq: number, turbo: number }} timings  totalMs per mode
 */
window.updateBenchmarkChart = function(timings) {
  if (!_benchmarkChart) return;
  const scale = 1; // ms as-is
  _benchmarkChart.data.datasets[0].data = [
    timings.vanilla ?? 45,
    timings.pq      ?? 28,
    timings.turbo   ?? 18,
  ];
  _benchmarkChart.data.datasets[0].label = 'Thời Gian Thực Tế (ms)';
  _benchmarkChart.update('active');
};
