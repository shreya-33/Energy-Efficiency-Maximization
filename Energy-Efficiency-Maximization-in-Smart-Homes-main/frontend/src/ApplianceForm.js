import React from 'react';
import './ApplianceForm.css'

const ApplianceForm = ({ index, applianceData, setApplianceData }) => {
  const handleInputChange = (event) => {
    const { name, value } = event.target;
    const updatedAppliances = [...applianceData];
    updatedAppliances[index] = { ...updatedAppliances[index], [name]: value };
    setApplianceData(updatedAppliances);
  };

  return (
    <div className='appliance-form '>
      <h3>Appliance {index + 1}</h3>
      <input
        type="text"
        name="name"
        placeholder="Name"
        onChange={handleInputChange}
      />
      <select name="type" onChange={handleInputChange}>
        <option value="NS">Non-Shiftable</option>
        <option value="PS">Power-Shiftable</option>
        <option value="TS">Time-Shiftable</option>
      </select>
      <input
        type="number"
        name="dissCoeff"
        placeholder="Diss. Coeff."
        onChange={handleInputChange}
      />
      <input
        type="text"
        name="powerRating"
        placeholder="Power Rating (kWh)"
        onChange={handleInputChange}
      />
      <input
        type="text"
        name="timeSlot"
        placeholder="Time Slot"
        onChange={handleInputChange}
      />
    </div>
  );
};

export default ApplianceForm;
