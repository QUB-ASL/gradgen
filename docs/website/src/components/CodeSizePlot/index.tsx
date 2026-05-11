import BrowserOnly from '@docusaurus/BrowserOnly';

const horizons = [
  20, 40, 100, 200, 300, 400, 500
];

const bicycleLOC = [
  1763, 3393, 8283, 16433, 24583, 32733, 40883
];

const casadiTimes = [
  5814, 11487, 28507, 56874, 85240, 113607, 141974
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

export default function CodeSizePlot() {
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
                y: bicycleLOC,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Bicycle model',
                line: { color: '#ca12a5', width: 3 },
                marker: { size: 7 },
              },
              {
                x: horizons,
                y: casadiTimes,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Larger model',
                line: { color: '#12b0be', width: 3 },
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
