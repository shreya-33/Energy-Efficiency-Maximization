// CostSavingsChart.js
import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend,
  } from 'chart.js';
  
  ChartJS.register(
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
  );
  
  //... rest of your component code
  

const CostSavingsChart = ({ ECwithoutDR, ECwithDR }) => {
  const data = {
    labels: ['Original Actions', 'Recommendations'],
    datasets: [
      {
        label: 'Electricity Cost',
        data: [ECwithoutDR, ECwithDR],
        backgroundColor: [
          'rgba(255, 99, 132, 0.2)',
          'rgba(54, 162, 235, 0.2)'
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)'
        ],
        borderWidth: 1
      }
    ]
  };

  return <Bar data={data} />;
};

export default CostSavingsChart;
