import React from 'react';
import './SubmitButton.css'

const SubmitButton = ({ handleSubmit }) => {
  return <button className = 'submit-button' onClick={handleSubmit}>Get Recommendations</button>;
};

export default SubmitButton;
