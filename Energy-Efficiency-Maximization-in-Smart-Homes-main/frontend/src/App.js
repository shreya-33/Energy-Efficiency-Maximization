import React, { useState } from 'react';
import NumberOfAppliancesForm from './NumberOfAppliancesForm';
import ApplianceForm from './ApplianceForm';
import SubmitButton from './SubmitButton';
import RecommendationsTable from './RecommendationsTable';
import CostSavingsChart from './CostSavingsChart';
import './App.css'

const App = () => {
  const [numberOfAppliances, setNumberOfAppliances] = useState(0);
  const [applianceData, setApplianceData] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [withoutDRActions, setWithoutDRActions] = useState([]);
  const [ECWithOutDR, setECWithOutDR] = useState(0);
  const [ECWithDR, setECWithDR] = useState(0);

  const handleSubmit = async (event) => {
    // Logic to collect all form data and send it to the backend

    console.log('Form data:', applianceData);
    const result = applianceData.map(obj => {
      return [obj.name, obj.type, parseFloat(obj.dissCoeff), parseFloat(obj.powerRating), JSON.parse(obj.timeSlot)];
    });
    event.preventDefault();

    fetch("http://127.0.0.1:5000/recommendations", {
      method: "POST",
      body: result,
    })
      .then((resp) => resp.json())
      .then((resp) => {
        // const recommendationsArray = Object.values(resp.recommendations);

        // // Convert without_DR_actions object to array
        // const withoutDRActionsArray = Object.values(resp.without_DR_actions);

        // console.log(recommendationsArray);
        // console.log(withoutDRActionsArray);

        setRecommendations(Object.values(resp.recommendations));
        setWithoutDRActions(Object.values(resp.without_DR_actions));
        setECWithOutDR(resp.ECWithoutDR);
        setECWithDR(resp.ECWithDR)
      })
      .catch((error) => {
        console.error("Error:", error);
      });

    // Add code here to send data to backend, e.g., using fetch API
  };

  return (
    <div className="app-container">
      <h2>Smart Home Energy Management</h2>
      <NumberOfAppliancesForm setNumberOfAppliances={setNumberOfAppliances} />
      {[...Array(numberOfAppliances)].map((_, index) => (
        <ApplianceForm
          key={index}
          index={index}
          applianceData={applianceData}
          setApplianceData={setApplianceData}
        />
      ))}
      <br />
      <SubmitButton handleSubmit={handleSubmit} />
      {recommendations.length >0 && <RecommendationsTable
        recommendations={recommendations}
        withoutDRActions={withoutDRActions}
      />}
      <br/><br/>
      {ECWithDR > 0 && <CostSavingsChart
        ECwithoutDR={ECWithOutDR}
        ECwithDR={ECWithDR}
      />

      }

      
      
    </div>
  );
};

export default App;
