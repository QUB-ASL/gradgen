import BrowserOnly from '@docusaurus/BrowserOnly';

const horizons = [
  20, 40, 60, 80, 100, 120, 140, 160, 180, 200
];

const gradgenTimes = [
  0.31, 0.61, 0.91, 1.21, 1.52, 1.82, 2.12, 2.43, 2.73, 3.03
];

const casadiTimes = [
  0.24, 0.5, 0.75, 0.98, 1.21, 1.46, 1.7, 1.94, 2.18, 2.43
];

const runtimeLayout = {
  autosize: true,
  margin: { l: 64, r: 24, t: 24, b: 60 },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  xaxis: {
    title: {
      text: 'Horizon N',
    },
    gridcolor: '#d9e2f2',
    zerolinecolor: '#d9e2f2',
    automargin: true,
  },
  yaxis: {
    title: {
      text: 'Average runtime (μs)',
    },
    gridcolor: '#d9e2f2',
    zerolinecolor: '#d9e2f2',
    automargin: true,
  },
  legend: {
    orientation: 'h',
    y: 1.18,
    x: 0,
  },
  font: {
    family: 'system-ui, sans-serif',
    size: 14,
    color: '#1f2937',
  },
};

const runtimeConfig = {
  responsive: true,
  displayModeBar: false,
};

export default function BenchmarkRuntimePlot() {
  return (
    <BrowserOnly fallback={<div>Loading runtime plot...</div>}>
      {() => {
        // Plotly is loaded only in the browser so Docusaurus can prerender safely.
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const Plotly = require('plotly.js-dist-min');
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const createPlotlyComponent = require('react-plotly.js/factory').default;
        const Plot = createPlotlyComponent(Plotly);

        return (
          <Plot
            data={[
              {
                x: horizons,
                y: gradgenTimes,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Gradgen',
                line: { color: '#0f766e', width: 3 },
                marker: { size: 7 },
              },
              {
                x: horizons,
                y: casadiTimes,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'CasADi',
                line: { color: '#b45309', width: 3 },
                marker: { size: 7 },
              },
            ]}
            layout={runtimeLayout}
            config={runtimeConfig}
            useResizeHandler
            style={{ width: '70%', height: '400px' }}
          />
        );
      }}
    </BrowserOnly>
  );
}
