import React from 'react';
import "./NumberOfAppliancesForm.css";

const NumberOfAppliancesForm = ({ setNumberOfAppliances }) => {
  return (
    <div className="number-of-appliances-form"> 
      <label>Number of Appliances: </label>
      <input
        type="number"
        onChange={(e) => setNumberOfAppliances(parseInt(e.target.value))}
      />
    </div>
  );
};

export default NumberOfAppliancesForm;
