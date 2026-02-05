// RecommendationsTable.js
import React from 'react';
import './RecommendationsTable.css'; // Import the CSS file for styling

const RecommendationsTable = ({ recommendations, withoutDRActions }) => {
  return (
    <table>
      <thead>
        <tr>
          <th>Time Slot</th>
          <th>Appliance Name</th>
          <th>Original Actions</th>
          <th>Recommended Actions</th>
        </tr>
      </thead>
      <tbody>
        {recommendations.map((slot, index) => {
          return Object.keys(slot).map((appliance) => {
            return (
              <tr key={`${index}-${appliance}`}>
                <td>{index + 1}</td>
                <td>{appliance}</td>
                <td>{withoutDRActions[index][appliance] || "None"}</td>
                <td>{slot[appliance] || "None"}</td>
              </tr>
            );
          });
        })}
      </tbody>
    </table>
  );
};

export default RecommendationsTable;
